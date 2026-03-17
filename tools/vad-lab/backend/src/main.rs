mod audio_source;
mod pipeline;
mod session;
mod ws;

use axum::{
    extract::WebSocketUpgrade,
    response::{Html, IntoResponse},
    routing::get,
    Router,
};
use clap::Parser;
use tower_http::cors::CorsLayer;

#[derive(Parser)]
#[command(name = "vad-lab", about = "VAD experimentation tool")]
struct Args {
    /// Port to listen on.
    #[arg(short, long, default_value = "3000")]
    port: u16,

    /// Host to bind to.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "vad_lab=info".into()),
        )
        .init();

    let args = Args::parse();

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/", get(index_handler))
        .layer(CorsLayer::permissive());

    let addr = format!("{}:{}", args.host, args.port);
    tracing::info!("vad-lab server listening on http://{addr}");
    tracing::info!("open http://{addr} in your browser");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn ws_handler(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(ws::handle_ws)
}

async fn index_handler() -> Html<&'static str> {
    Html(
        r#"<!DOCTYPE html>
<html>
<head><title>vad-lab</title></head>
<body>
    <h1>vad-lab</h1>
    <p>Frontend not yet built. Run <code>npm run dev</code> in <code>tools/vad-lab/frontend/</code> for the dev server.</p>
    <p>WebSocket endpoint: <code>ws://localhost:3000/ws</code></p>
</body>
</html>"#,
    )
}
