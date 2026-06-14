use crate::errors::GatewayError;
use mlx_runtime_protocol::{decode_worker_message, encode_worker_message, WorkerMessage};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;

/// Sends a worker message over a Unix stream.
#[allow(dead_code)]
pub fn send_message(stream: &mut UnixStream, message: &WorkerMessage) -> Result<(), GatewayError> {
    let encoded = encode_worker_message(message);
    writeln!(stream, "{encoded}")?;
    stream.flush()?;
    Ok(())
}

/// Reads a single worker message from a Unix stream.
#[allow(dead_code)]
pub fn read_message(stream: &UnixStream) -> Result<Option<WorkerMessage>, GatewayError> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    let bytes = reader.read_line(&mut line)?;
    if bytes == 0 {
        return Ok(None);
    }
    Ok(decode_worker_message(&line))
}
