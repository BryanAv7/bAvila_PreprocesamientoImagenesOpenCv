// Parte 1C: Preprocesamiento de imágenes OpenCV(CPU vs GPU)
// Nombre: Bryan Avila
// Carrera: Computación
// Materia: Visión por Computadora
// Fecha: 2025-06-28

// Librerías necesarias por utilizar
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <chrono>
#include <iomanip>
#include <filesystem>

// Espacio de nombres
using namespace cv;
using namespace std;
using namespace std::chrono;
namespace fs = std::filesystem;

// ------------------------------------------------------------------------------------------
// Función: procesoCPU
// Descripción: Aplicar operaciones de OpenCV a las imágenes que seran cargadas previamente, ejecutandose en la CPU, incluyendo:
//              - Conversión a escala de grises
//              - Suavizado con filtro Gaussiano
//              - Erosión y dilatación
//              - Detección de bordes (Canny)
//              - Ecualización de histograma
// ------------------------------------------------------------------------------------------
Mat procesoCPU(const Mat& frame) {
    Mat gray, blurred, eroded, dilated, edges, equalized;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(5, 5), 1.5);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    erode(blurred, eroded, kernel);
    dilate(eroded, dilated, kernel);

    Canny(dilated, edges, 50, 150);
    equalizeHist(edges, equalized);

    return equalized;
}

// ------------------------------------------------------------------------------------------
// Función: procesoGPU
// Descripción: Aplica un pipeline de procesamiento de imágenes en GPU usando CUDA, incluyendo:
//              - Conversión a escala de grises
//              - Filtro Gaussiano
//              - Erosión y dilatación morfológica
//              - Detección de bordes con Canny
//              - Ecualización de histograma
// ------------------------------------------------------------------------------------------
vector<Mat> procesoGPU(const vector<Mat>& frames,
                       Ptr<cuda::Filter>& gauss,
                       Ptr<cuda::Filter>& erodeF,
                       Ptr<cuda::Filter>& dilateF,
                       Ptr<cuda::CannyEdgeDetector>& canny) {
    vector<Mat> resultados;
    resultados.reserve(frames.size());

    // Subir todas las imágenes a GPU
    vector<cuda::GpuMat> d_frames(frames.size());
    for (size_t i = 0; i < frames.size(); ++i) {
        d_frames[i].upload(frames[i]);
    }

    // Procesar todas las imágenes en GPU
    for (size_t i = 0; i < d_frames.size(); ++i) {
        cuda::GpuMat d_gray, d_blur, d_eroded, d_dilated, d_edges, d_equalized;

        cuda::cvtColor(d_frames[i], d_gray, COLOR_BGR2GRAY);
        gauss->apply(d_gray, d_blur);
        erodeF->apply(d_blur, d_eroded);
        dilateF->apply(d_eroded, d_dilated);
        canny->detect(d_dilated, d_edges);
        cuda::equalizeHist(d_edges, d_equalized);

        d_frames[i] = d_equalized; // guardar el resultado en GPU
    }

    // Descargar todos los resultados a CPU
    for (const auto& d_res : d_frames) {
        Mat resultado;
        d_res.download(resultado);
        resultados.push_back(resultado);
    }

    return resultados;
}

// ------------------------------------------------------------------------------------------
// Función principal
// Descripción: Lee imágenes desde un directorio, aplica procesamiento con CPU y GPU,
//              guarda los resultados y compara el tiempo de ejecución.
// ------------------------------------------------------------------------------------------
int main() {
    // Directorio de imágenes
    string rutaCarpeta = "/home/bryan/Documentos/segundoInterciclo/TrabajoU4PartC/imagenes/";
    vector<String> archivos;
    glob(rutaCarpeta + "*.jpg", archivos);

    if (archivos.empty()) {
        cerr << "No se encontraron imágenes." << endl;
        return -1;
    }

    // Crear directorios de resultados
    fs::create_directories("resultados/cpu");
    fs::create_directories("resultados/gpu");

    // Cargar imágenes
    vector<Mat> imagenes;
    for (const auto& archivo : archivos) {
        Mat img = imread(archivo);
        if (!img.empty()) imagenes.push_back(img);
    }

    // Procesamiento CPU
    auto startCPU = high_resolution_clock::now();
    for (size_t i = 0; i < imagenes.size(); ++i) {
        Mat resCPU = procesoCPU(imagenes[i]);
        string nombre = fs::path(archivos[i]).filename().string();
        imwrite("resultados/cpu/" + nombre, resCPU);
    }
    auto endCPU = high_resolution_clock::now();
    long long tiempoCPU = duration_cast<milliseconds>(endCPU - startCPU).count();

    // Crear filtros de CUDA
    Mat dummyGray;
    cvtColor(imagenes[0], dummyGray, COLOR_BGR2GRAY);
    cuda::GpuMat d_dummyGray(dummyGray);

    Ptr<cuda::Filter> gauss = cuda::createGaussianFilter(d_dummyGray.type(), d_dummyGray.type(), Size(5, 5), 1.5);
    Ptr<cuda::Filter> erodeF = cuda::createMorphologyFilter(MORPH_ERODE, d_dummyGray.type(), getStructuringElement(MORPH_RECT, Size(5, 5)));
    Ptr<cuda::Filter> dilateF = cuda::createMorphologyFilter(MORPH_DILATE, d_dummyGray.type(), getStructuringElement(MORPH_RECT, Size(5, 5)));
    Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector(50, 150);

    // Procesamiento GPU
    auto startGPU = high_resolution_clock::now();
    vector<Mat> resultadosGPU = procesoGPU(imagenes, gauss, erodeF, dilateF, canny);
    auto endGPU = high_resolution_clock::now();
    long long tiempoGPU = duration_cast<milliseconds>(endGPU - startGPU).count();

    // Guardar resultados GPU
    for (size_t i = 0; i < resultadosGPU.size(); ++i) {
        string nombre = fs::path(archivos[i]).filename().string();
        imwrite("resultados/gpu/" + nombre, resultadosGPU[i]);
    }

    // Mostrar resultados del procesamiento
    cout << fixed << setprecision(2);
    cout << "\n===============================\n";
    cout << "Tiempo total CPU: " << tiempoCPU << " ms\n";
    cout << "Tiempo total GPU: " << tiempoGPU << " ms\n";
    if (tiempoGPU > 0) {
        cout << "Speedup total: " << (double)tiempoCPU / tiempoGPU << "x\n";
    }
    cout << "===============================\n";

    return 0;
}
