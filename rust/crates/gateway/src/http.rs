use crate::config::ServerConfig;
use crate::errors::GatewayError;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;

/// Serves the minimal health endpoint.
pub fn serve(server: &ServerConfig, healthy: Arc<AtomicBool>) -> Result<(), GatewayError> {
    let listener = TcpListener::bind(format!("{}:{}", server.host, server.port))?;

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let healthy = healthy.clone();
                thread::spawn(move || {
                    let _ = handle_connection(stream, healthy);
                });
            }
            Err(err) => return Err(GatewayError::Io(err)),
        }
    }

    Ok(())
}

fn handle_connection(mut stream: TcpStream, healthy: Arc<AtomicBool>) -> Result<(), GatewayError> {
    let mut request_line = String::new();
    {
        let mut reader = BufReader::new(&stream);
        let _ = reader.read_line(&mut request_line)?;
    }

    let (status, body) = response_for_request_line(&request_line, healthy.load(Ordering::SeqCst));

    write_response(&mut stream, status, body)?;
    Ok(())
}

fn response_for_request_line(request_line: &str, healthy: bool) -> (&'static str, &'static str) {
    if request_line.starts_with("GET /health ") {
        if healthy {
            ("200 OK", "healthy")
        } else {
            ("503 Service Unavailable", "unhealthy")
        }
    } else {
        ("404 Not Found", "not found")
    }
}

fn write_response(stream: &mut TcpStream, status: &str, body: &str) -> Result<(), GatewayError> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nContent-Type: text/plain; charset=utf-8\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )?;
    stream.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_endpoint_returns_503_when_worker_is_not_ready() {
        assert_eq!(
            response_for_request_line("GET /health HTTP/1.1\r\n", false),
            ("503 Service Unavailable", "unhealthy")
        );
    }

    #[test]
    fn health_endpoint_returns_200_when_worker_is_ready() {
        assert_eq!(
            response_for_request_line("GET /health HTTP/1.1\r\n", true),
            ("200 OK", "healthy")
        );
    }
}
