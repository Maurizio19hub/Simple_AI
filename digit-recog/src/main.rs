use axum::response::IntoResponse;
use axum::{response::Html, routing::get, routing::post, Router, Json};
use std::result;
use std::{net::SocketAddr, ops::Mul};
use axum_extra::extract::{multipart, Multipart};
use image::{DynamicImage, GenericImageView, GrayAlphaImage, ImageBuffer, Rgba};
use serde::Serialize;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple, PyList};

#[tokio::main]
async fn main() {
    let app = Router::new()
            .route("/", get(homepage))
            .route("/upload", post(upload));

    //Creo il soket (ip + porta)
    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    println!("Server in ascolto su {}", addr.to_string());
    //Creo listener tcp, con riferimento alla variabile addr
    //await aspetta che termini l'operazione
    //unwrap per la gestione degli errori
    let listener  = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
    
}

async fn upload(mut multipart: Multipart) -> impl IntoResponse {
    while let Some(field) = multipart.next_field().await.unwrap() {
        if let Some(name) = field.file_name() {
            println!("File ricevuto: {}", name);

            // 1. Estraggo i byte dell’immagine
            let data = field.bytes().await.unwrap();

            // 2. Decodifico l’immagine
            let img = image::load_from_memory(&data).unwrap();

            // 3. Ridimensiono a 28x28 (MNIST size)
            let resized = img.resize_exact(28, 28, image::imageops::FilterType::Nearest);

            // 4. Compositing su bianco (elimina trasparenza)
            let rgba = resized.to_rgba8();
            let (w, h) = rgba.dimensions();
            let mut composited = ImageBuffer::from_pixel(w, h, Rgba([255, 255, 255, 255]));
            for (x, y, p) in rgba.enumerate_pixels() {
                let alpha = p[3] as f32 / 255.0;
                let r = (p[0] as f32 * alpha + 255.0 * (1.0 - alpha)).round() as u8;
                let g = (p[1] as f32 * alpha + 255.0 * (1.0 - alpha)).round() as u8;
                let b = (p[2] as f32 * alpha + 255.0 * (1.0 - alpha)).round() as u8;
                composited.put_pixel(x, y, Rgba([r, g, b, 255]));
            }

            // 5. Scala di grigi
            let gray = DynamicImage::ImageRgba8(composited).to_luma8();
            gray.save("debug.png");
            // 6. Inverto i colori (MNIST è bianco su nero)
            let inverted: Vec<u8> = gray.pixels().map(|p| 255 - p[0]).collect();

            // 7. Normalizzo a [0,1] (float) come input per NN
            let normalized: Vec<f32> = inverted.iter().map(|&v| v as f32 / 255.0).collect();
            let prediction = callPythonAlg(normalized).unwrap_or("errore".to_string());
            println!("{} ciaoaosofo", prediction);
            return Json(serde_json::json!({ "result": prediction }));
            
            
            //return Json(MnistLike { pixels: inverted });
        }
    }
    return Json(serde_json::json!({ "result": "none" }));
}


fn callPythonAlg(arg1: Vec<f32>) -> PyResult<String> {
    Python::with_gil(|py| {
        println!("=== DEBUG PYTHON IMPORT ===");
        
        // Debug: controlla sys.path
        let sys = PyModule::import(py, "sys")?;
        let path: &PyList = sys.getattr("path")?.downcast()?;
        path.insert(0, "/home/maurizio/.conda/envs/mnist-nn/lib/python3.10/site-packages")?;
        path.insert(0, "/home/maurizio/.conda/envs/mnist-nn/lib/python3.10")?;
        path.insert(0, "/home/maurizio/AI_try")?;
        
        println!("sys.path aggiornato:");
        for (i, p) in path.iter().enumerate() {
            println!("  [{}]: {:?}", i, p);
        }

       
        
        // Debug: controlla se il file esiste
        let module_path = "/home/maurizio/AI_try/python_module_for_rust.py";
        println!("File esiste: {}", std::path::Path::new(module_path).exists());
        
        // Debug: prova l'importazione con gestione errore specifica
        println!("Tentativo importazione modulo...");
        match PyModule::import(py, "python_module_for_rust") {
            Ok(module) => {
                println!("✅ Modulo importato correttamente");
                
                // Continua con la chiamata alla funzione...
                let func = module.getattr("testingFun")?;
                let args = PyTuple::new(py, &[arg1.into_py(py)]);
                let result = func.call1(args)?;
                let ret: String = result.extract()?;
                
                Ok(ret)
            },
            Err(e) => {
                println!("❌ Errore importazione: {:?}", e);
                println!("Tipo errore: {}", e);
                Err(e)
            }
        }
    })
}


//&'static str : valore di ritorno è il riferimento a una stringa statica
async fn homepage() -> Html<String> {
    let html_content = std::fs::read_to_string("index.html").unwrap();
    Html(html_content) //return senza ;
}


#[derive(Serialize)]
struct MnistLike {
    pixels: Vec<u8>, // 784 valori
}
