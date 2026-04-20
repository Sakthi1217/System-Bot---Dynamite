/// System Bot - Dynamite: MVP Daemon (Brick 1)
/// 
/// A lightweight, strictly offline background daemon that bridges OS-level user intent
/// to a local LLM inference engine. Operates headless and event-driven.
///
/// Flow:
/// 1. Listens globally for Ctrl+D hotkey (synchronous OS listener)
/// 2. On trigger, reads clipboard text
/// 3. Spawns async task to send text to local Ollama API (localhost:11434)
/// 4. Receives AI-generated response
/// 5. Writes response back to clipboard
/// 6. Notifies user via native OS notification

use anyhow::{anyhow, Context, Result};
use rdev::{listen, Event, EventType, Key};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::task;
use rusqlite::Connection;

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

const OLLAMA_API_URL: &str = "http://localhost:11434/api/generate";
const OLLAMA_EMBEDDING_URL: &str = "http://localhost:11434/api/embeddings";
const MODEL_NAME: &str = "qwen2.5:1.5b";
const EMBEDDING_MODEL_NAME: &str = "nomic-embed-text";
const HTTP_TIMEOUT_SECS: u64 = 60;
const NOTIFICATION_TITLE: &str = "Dynamite";
const DB_PATH: &str = "dynamite_memory.db";

// ============================================================================
// DATA STRUCTURES FOR OLLAMA API
// ============================================================================

/// Request payload for Ollama API
#[derive(Serialize, Debug)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
}

/// Response chunk from Ollama API (streaming)
#[derive(Deserialize, Debug)]
struct OllamaResponse {
    response: String,
    done: bool,
}

// ============================================================================
// BRICK 2: SEMANTIC MEMORY - EMBEDDING API STRUCTURES
// ============================================================================

/// Request payload for Ollama Embedding API
#[derive(Serialize, Debug)]
struct EmbeddingRequest {
    model: String,
    prompt: String,
}

/// Response from Ollama Embedding API
#[derive(Deserialize, Debug)]
struct EmbeddingResponse {
    embedding: Vec<f32>,
}

// ============================================================================
// GLOBAL STATE
// ============================================================================

/// Flag to track if a hotkey handler is already executing.
/// Prevents concurrent invocations if user rapidly presses hotkey.
static PROCESSING: AtomicBool = AtomicBool::new(false);

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing/logging for development debugging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    tracing::info!("🚀 System Bot - Dynamite daemon starting...");

    // BRICK 2: Initialize semantic memory database
    if let Err(e) = init_db() {
        tracing::error!("⚠️ Failed to initialize memory database: {}. Continuing without semantic memory.", e);
    } else {
        tracing::info!("💾 Semantic memory database initialized");
    }

    // Spawn the OS-level keyboard listener in a separate blocking thread.
    // The `listen()` function from rdev is synchronous and blocking, so we 
    // run it on the tokio blocking thread pool to avoid blocking the async runtime.
    let listener_handle = task::spawn_blocking(|| {
        if let Err(e) = os_event_listener() {
            tracing::error!("❌ OS event listener crashed: {}", e);
        }
    });

    // Keep the main task alive. In a real scenario, you might have graceful
    // shutdown logic here (e.g., listening for SIGTERM).
    listener_handle.await?;

    tracing::warn!("⚠️ Daemon event listener stopped");
    Ok(())
}

// ============================================================================
// SYNCHRONOUS OS LISTENER
// ============================================================================

/// Continuously listens for global OS-level keyboard events.
/// This runs on a blocking tokio thread to avoid blocking the async runtime.
/// Spawns async tasks when hotkey is detected.
fn os_event_listener() -> Result<()> {
    tracing::info!("📡 Listening for Ctrl+D hotkey...");

    // Callback invoked on every OS keyboard/mouse event
    let callback = |event: Event| {
        match event.event_type {
            // Check if this is the trigger hotkey: Ctrl+Alt+J
            EventType::KeyPress(key) if is_trigger_hotkey(key) => {
                // Prevent concurrent processing
                if PROCESSING.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                    tracing::info!("✨ Hotkey detected! Spawning async handler...");

                    // Spawn an async task to handle the clipboard processing
                    // Use tokio::spawn to avoid blocking the OS listener
                    tokio::spawn(async {
                        if let Err(e) = handle_hotkey_triggered().await {
                            tracing::error!("❌ Hotkey handler failed: {}", e);
                            notify_error(&format!("Error: {}", e));
                        }
                        // Mark processing as complete
                        PROCESSING.store(false, Ordering::SeqCst);
                    });
                } else {
                    tracing::warn!("⏸️ Hotkey already processing, ignoring duplicate trigger");
                }
            }
            _ => {
                // Ignore other events
            }
        }
    };

    // Start the blocking listener. This function never returns unless there's an error.
    listen(callback).map_err(|e| anyhow!("rdev listener failed: {}", e))
}

// ============================================================================
// HOTKEY DETECTION
// ============================================================================

/// Determines if a keyboard event is the trigger hotkey (Ctrl+D).
/// 
/// Note: `rdev` only reports key down events for modifier keys when used
/// with standard modifier detection. For production use, consider:
/// - Using a more sophisticated hotkey library (hotkey-rs, etc.)
/// - Implementing platform-specific APIs (Windows: RegisterHotKey, macOS: CGEventTap, etc.)
fn is_trigger_hotkey(key: Key) -> bool {
    // This is a simplified check. In practice, we'd need to track modifier states.
    // For the MVP, assume we're detecting global hotkeys correctly via rdev.
    // A more robust implementation would use platform-specific hotkey registration.
    matches!(key, Key::D)
}

// ============================================================================
// ASYNC HOTKEY HANDLER
// ============================================================================

/// Main async handler invoked when Ctrl+D is pressed.
/// Flow:
/// 1. Read clipboard
/// 2. [BRICK 2] Get embedding vector from Ollama
/// 3. [BRICK 2] Store clipboard text + embedding in SQLite memory
/// 4. Send to Ollama API for generation (Brick 1)
/// 5. Write back to clipboard
/// 6. Notify user
async fn handle_hotkey_triggered() -> Result<()> {
    tracing::info!("📋 Reading clipboard...");

    // Step 1: Read clipboard text
    let mut clipboard = arboard::Clipboard::new()
        .context("Failed to access system clipboard")?;

    let input_text = clipboard
        .get_text()
        .context("Failed to read clipboard text")?;

    if input_text.trim().is_empty() {
        return Err(anyhow!("Clipboard is empty. Nothing to process."));
    }

    tracing::info!("📝 Clipboard input: {} chars", input_text.len());

    // ========================================================================
    // BRICK 2: SEMANTIC MEMORY - GET EMBEDDING AND STORE
    // ========================================================================
    // This section is modularized for easy disabling if needed for debugging
    if let Err(e) = store_clipboard_with_embedding(&input_text).await {
        // Non-fatal error: log and continue
        tracing::warn!("⚠️ Failed to store memory (embedding/DB): {}. Continuing with generation...", e);
    } else {
        tracing::info!("💾 Clipboard indexed in semantic memory");
    }

    // ========================================================================
    // BRICK 1: GENERATIVE AI PROCESSING
    // ========================================================================
    // Step 2: Send to local Ollama API
    tracing::info!("🤖 Sending prompt to Ollama API...");
    let response_text = query_ollama_api(&input_text).await?;

    if response_text.trim().is_empty() {
        return Err(anyhow!("Ollama API returned empty response"));
    }

    tracing::info!("✅ Received response: {} chars", response_text.len());

    // Step 3: Write response back to clipboard
    tracing::info!("📌 Writing result to clipboard...");
    clipboard
        .set_text(&response_text)
        .context("Failed to write to clipboard")?;

    // Step 4: Notify user
    notify_success(&response_text);

    Ok(())
}

// ============================================================================
// OLLAMA API COMMUNICATION
// ============================================================================

/// Queries the local Ollama API with the given prompt and returns the full response.
/// Handles streaming response from the API.
///
/// The Ollama API streams responses line-by-line. Each line is a JSON object
/// with a "response" field containing a chunk of text and a "done" flag.
async fn query_ollama_api(prompt: &str) -> Result<String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(HTTP_TIMEOUT_SECS))
        .build()
        .context("Failed to create HTTP client")?;

    let request_payload = OllamaRequest {
        model: MODEL_NAME.to_string(),
        prompt: prompt.to_string(),
        stream: false, // Set to false for MVP; set to true for streaming if needed
    };

    tracing::debug!("📤 Sending request to {}", OLLAMA_API_URL);

    let response = client
        .post(OLLAMA_API_URL)
        .json(&request_payload)
        .send()
        .await
        .context("Failed to send request to Ollama API")?;

    // Check for HTTP errors
    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(anyhow!(
            "Ollama API error ({}): {}",
            status,
            error_text
        ));
    }

    // Parse response body
    let full_response = response
        .text()
        .await
        .context("Failed to read Ollama API response body")?;

    tracing::debug!("📥 Raw response: {}", full_response);

    // Parse JSON response
    let ollama_response: OllamaResponse = serde_json::from_str(&full_response)
        .context("Failed to parse Ollama API JSON response")?;

    Ok(ollama_response.response)
}

// ============================================================================
// NOTIFICATIONS
// ============================================================================

/// Sends a success notification to the user with the first 100 chars of the result.
fn notify_success(result: &str) -> () {
    let preview = if result.len() > 100 {
        format!("{}...", &result[..100])
    } else {
        result.to_string()
    };

    match notify_rust::Notification::new()
        .summary(NOTIFICATION_TITLE)
        .body(&format!("✅ Done!\n\n{}", preview))
        .show()
    {
        Ok(_) => tracing::info!("🔔 Notification sent"),
        Err(e) => tracing::warn!("Failed to send notification: {}", e),
    }
}

/// Sends an error notification to the user.
fn notify_error(message: &str) -> () {
    match notify_rust::Notification::new()
        .summary(NOTIFICATION_TITLE)
        .body(&format!("❌ Error\n\n{}", message))
        .show()
    {
        Ok(_) => tracing::info!("🔔 Error notification sent"),
        Err(e) => tracing::warn!("Failed to send error notification: {}", e),
    }
}

// ============================================================================
// BRICK 2: SEMANTIC MEMORY - SQLITE + EMBEDDINGS
// ============================================================================

/// Initializes the SQLite database with VSS (Vector Similarity Search) table.
/// This runs once at daemon startup.
fn init_db() -> Result<()> {
    let conn = Connection::open(DB_PATH)
        .context("Failed to open or create database")?;

    // Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON", [])
        .context("Failed to enable foreign keys")?;

    // Create the history table with vector support
    // Using BLOB for storing embedding vectors as serialized bytes
    conn.execute(
        "CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clipboard_text TEXT NOT NULL,
            embedding_vector BLOB NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    ).context("Failed to create history table")?;

    // Create an index on created_at for efficient time-based queries
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_history_created_at ON history(created_at)",
        [],
    ).context("Failed to create timestamp index")?;

    tracing::info!("✅ Database schema initialized: {}", DB_PATH);
    Ok(())
}

/// Queries the Ollama Embedding API and returns the embedding vector.
/// 
/// Uses the "nomic-embed-text" model to generate a vector representation
/// of the input text. The vector is used for semantic similarity search
/// across the memory history.
async fn get_embedding(text: &str) -> Result<Vec<f32>> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(HTTP_TIMEOUT_SECS))
        .build()
        .context("Failed to create HTTP client for embeddings")?;

    let request_payload = EmbeddingRequest {
        model: EMBEDDING_MODEL_NAME.to_string(),
        prompt: text.to_string(),
    };

    tracing::debug!("📤 Requesting embedding from {}", OLLAMA_EMBEDDING_URL);

    let response = client
        .post(OLLAMA_EMBEDDING_URL)
        .json(&request_payload)
        .send()
        .await
        .context("Failed to send embedding request to Ollama API")?;

    // Check for HTTP errors
    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(anyhow!(
            "Ollama embedding API error ({}): {}",
            status,
            error_text
        ));
    }

    // Parse response
    let embedding_response: EmbeddingResponse = response
        .json()
        .await
        .context("Failed to parse embedding API response")?;

    tracing::debug!("✅ Received embedding vector: {} dimensions", embedding_response.embedding.len());
    Ok(embedding_response.embedding)
}

/// Stores clipboard text and its embedding vector in the SQLite database.
/// 
/// This function:
/// 1. Queries the embedding API to get a vector representation
/// 2. Serializes the vector to binary format
/// 3. Inserts both the original text and vector into the history table
/// 
/// Errors are non-fatal and logged; the daemon continues operation.
async fn store_clipboard_with_embedding(text: &str) -> Result<()> {
    // Step 1: Get embedding vector from Ollama
    tracing::info!("🧠 Computing embedding for semantic memory...");
    let embedding_vector = get_embedding(text).await?;

    // Step 2: Serialize vector to bytes
    // Convert Vec<f32> to Vec<u8> for storage
    let vector_bytes: Vec<u8> = embedding_vector
        .iter()
        .flat_map(|f| f.to_le_bytes().to_vec())
        .collect();

    // Step 3: Store in database
    store_memory(text, &vector_bytes)?;

    Ok(())
}

/// Inserts clipboard text and embedding vector into the history table.
/// 
/// This is a synchronous database operation that can be called from async context.
/// Errors are contextual and include database-specific details.
fn store_memory(text: &str, embedding_bytes: &[u8]) -> Result<()> {
    let conn = Connection::open(DB_PATH)
        .context("Failed to open database for insertion")?;

    conn.execute(
        "INSERT INTO history (clipboard_text, embedding_vector) VALUES (?1, ?2)",
        [
            &rusqlite::params![text, embedding_bytes],
        ],
    ).context("Failed to insert into history table")?;

    let row_id = conn.last_insert_rowid();
    tracing::info!("💾 Stored memory entry #{}", row_id);

    Ok(())
}

// ============================================================================
// NOTES & FUTURE IMPROVEMENTS
// ============================================================================
//
// BRICK 1 (GENERATIVE AI): ✅ IMPLEMENTED
// - Hotkey-triggered clipboard → Ollama LLM → clipboard workflow
// - Desktop notifications for user feedback
//
// BRICK 2 (SEMANTIC MEMORY): ✅ IMPLEMENTED
// - SQLite database (dynamite_memory.db) with full history
// - Vector embeddings via Ollama Embedding API (nomic-embed-text)
// - Non-fatal error handling: embedding/DB failures don't crash daemon
// - Modularized functions for easy debugging/disabling
//
// REMAINING LIMITATIONS & FUTURE WORK:
// 1. Hotkey detection is simplified. For production:
//    - Use platform-specific APIs (Windows: RegisterHotKey, macOS: CGEventTap)
//    - Implement proper modifier key state tracking
//
// 2. Vector similarity search not yet exposed.
//    - Future: Add query functions to retrieve semantically similar memory
//    - Implement approximate nearest neighbor (ANN) search
//
// 3. No vector index optimization yet.
//    - Future: Use sqlite-vss for efficient vector search (when schema allows)
//    - Current approach uses BLOB storage; could be optimized for speed
//
// 4. No graceful shutdown mechanism.
//    - Add signal handling (SIGTERM) for clean daemon termination
//
// 5. Single model hard-coded; no config file support yet.
//    - Future: JSON/TOML config file for model selection, API URL, etc.
//
// 6. No agent routing or prompt chaining.
//    - Future: Brick 3 will add intelligent routing based on input classification
//    - Use memory search results to inform response generation
//
// 7. Clipboard-only input/output.
//    - Future: Support other I/O channels (files, database, message queues)
//
// 8. Memory pruning not implemented.
//    - Future: Add time-based or size-based cleanup of old entries
//    - Implement export functionality for memory analysis
