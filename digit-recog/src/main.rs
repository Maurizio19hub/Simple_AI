use axum::{response::Html, routing::get, routing::post, Router, Json};
use std::{net::SocketAddr, ops::Mul};
use axum_extra::extract::{multipart, Multipart};
use image::{DynamicImage, GenericImageView, GrayAlphaImage, ImageBuffer, Rgba};
use serde::Serialize;

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

async fn upload(mut multipart: Multipart) -> Json<MnistLike> {
    while let Some(field) = multipart.next_field().await.unwrap() {
        if let Some(name) = field.file_name() {
            println!("File ricevuto : {}", name);

            //Estraggo i byte dell'immagine
            let data = field.bytes().await.unwrap();
            //decodifica da formato immagine a DynamicImage
            let img = image::load_from_memory(&data).unwrap();

            //Ridimensionameto e scala di grigi
            let resized = img.resize_exact(28, 28, image::imageops::FilterType::Nearest);
            
            //Composit su bianco
            let rgba = resized.to_rgba8();
            let (w, h) = rgba.dimensions();
            let mut composited = ImageBuffer::from_pixel(w, h, Rgba([255, 255, 255, 255]));
            for (x, y, p) in rgba.enumerate_pixels() {
                let alpha = p[3] as f32 / 255.0;
                let r = (p[0] as f32 * alpha + 255.0 * (1.0 - alpha)).round() as u8;
                let g = (p[1] as f32 * alpha + 255.0 * (1.0 - alpha)).round() as u8;
                let b = (p[2] as f32 * alpha + 255.0 * (1.0 - alpha)).round() as u8;
                composited.put_pixel(x, y, Rgba([r,g,b,255]));
            }

            
            let gray = DynamicImage::ImageRgba8(composited).to_luma8();

            for p in gray.pixels() {
                println!("gray: {}", p[0])
            }
            //Inversione colori, mnist è bianco su nero e non nero su bianco come il canva 
            let inverted : Vec<u8> = gray.pixels().map(|p|255 - p[0]).collect();

            return Json(MnistLike { pixels: inverted });
        }
    }
    return Json(MnistLike { pixels: vec![] });
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
