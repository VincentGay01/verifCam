#define GLM_ENABLE_EXPERIMENTAL
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <nlohmann/json.hpp>
#include <regex>
using namespace glm;
using namespace cv;
using namespace std;
using json = nlohmann::json;
std::string err, warn;
std::vector<float> texCoords;
std::vector<float> normals;
std::vector<int> texNumbers;
std::vector<std::string> textureFiles;


// Variables globales pour la caméra

glm::mat4 alignmentMatrix(1.0f); // Matrice d'alignement pour le plan d'overlay
std::vector<float> vertices; // Variables pour le maillage
std::vector<unsigned int> indices; // Indices du maillage
// Variables pour le plan d'overlay
GLuint overlayVAO, overlayVBO, overlayEBO;
GLuint overlayTexture;
GLuint overlayShaderProgram;
GLuint shaderProgram;
GLuint texCoordVBO;
GLuint textureID;
GLuint planProgram;
glm::vec3 cameraPos = glm::vec3(0, 0, 0.5);
glm::mat4 view, viewMatrix, projectionMatrix;
float overlayZ = -0.5f; // Distance entre la caméra et le plan d'overlay 


// Matrices de vue et de projection
glm::mat4 projection = glm::perspective(glm::radians(60.f), 1.0f, 0.1f, 1000.0f);
glm::mat4 Model;
glm::mat4 MVP;
// Callback pour ajuster la taille de la fenêtre
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}


struct Position {
    double x, y, z;
};
struct CameraParams {
    double focal_length_x;
    double focal_length_y;
    double principal_point_x;
    double principal_point_y;
    int image_width;   // Nécessaire pour la conversion en projection OpenGL
    int image_height;  // Nécessaire pour la conversion en projection OpenGL
};


// Fonction pour extraire la position à partir du nom
Position extractPositionFromName(const std::string& name) {
    Position pos{ 0.0, 0.0, 0.0 };

    // Utiliser une expression régulière pour extraire les coordonnées
    std::regex posPattern(R"(Pos: \(([0-9.]+), ([0-9.]+), ([0-9.]+)\))");
    std::smatch matches;

    if (std::regex_search(name, matches, posPattern) && matches.size() == 4) {
        // Conversion des chaînes capturées en nombres
        pos.x = std::stod(matches[1].str());
        pos.y = std::stod(matches[2].str());
        pos.z = std::stod(matches[3].str());
    }
    return pos;
}

// Fonction pour compiler un shader
GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    // Vérification des erreurs de compilation
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Erreur de compilation du shader : " << infoLog << std::endl;
    }
    return shader;
}

// Fonction pour créer un programme shader
GLuint createShaderProgram(const char* vertexSource, const char* fragmentSource) {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSource);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Vérification des erreurs de liaison
    GLint success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cerr << "Erreur de liaison du programme shader : " << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

bool loadPLY_clean(const std::string& filename,
    std::vector<float>& vertices,
    std::vector<unsigned int>& indices,
    std::vector<float>& normals,
    std::vector<float>& texCoords,
    std::vector<std::string>& textureFiles) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier " << filename << std::endl;
        return false;
    }

    std::string line;
    bool headerEnded = false;
    int vertexCount = 0, faceCount = 0;
    bool hasNormals = false;
    bool hasTexCoordList = false;

    // Temporaire pour lire les sommets et normales
    std::vector<glm::vec3> tempPositions;
    std::vector<glm::vec3> tempNormals;

    // Lire header
    while (std::getline(file, line)) {
        if (line == "end_header") {
            headerEnded = true;
            break;
        }
        std::istringstream iss(line);
        std::string word;
        iss >> word;
        if (word == "element") {
            std::string type;
            int count;
            iss >> type >> count;
            if (type == "vertex") vertexCount = count;
            if (type == "face") faceCount = count;
        }
        else if (word == "property" && line.find("nx") != std::string::npos) {
            hasNormals = true;
        }
        else if (word == "property list" && line.find("texcoord") != std::string::npos) {
            hasTexCoordList = true;
        }
        else if (word == "comment" && line.find("TextureFile") != std::string::npos) {
            size_t pos = line.find("TextureFile");
            if (pos != std::string::npos) {
                std::string texturePath = line.substr(pos + 11);
                texturePath.erase(0, texturePath.find_first_not_of(" \t"));
                texturePath.erase(texturePath.find_last_not_of(" \t") + 1);
                textureFiles.push_back(texturePath);
            }
        }
    }

    if (!headerEnded) {
        std::cerr << "Erreur : fin de header non trouvée" << std::endl;
        return false;
    }

    // Lire les vertices
    for (int i = 0; i < vertexCount; ++i) {
        std::getline(file, line);
        std::istringstream iss(line);
        float x, y, z, nx = 0.0f, ny = 0.0f, nz = 0.0f;
        iss >> x >> y >> z;
        if (hasNormals) {
            iss >> nx >> ny >> nz;
        }
        tempPositions.emplace_back(x, y, z);
        tempNormals.emplace_back(nx, ny, nz);
    }

    // Lire les faces avec texcoord list
    for (int i = 0; i < faceCount; ++i) {
        std::getline(file, line);
        std::istringstream iss(line);

        int nVerts;
        iss >> nVerts;
        if (nVerts != 3) continue; // on ignore les non-triangles

        int i0, i1, i2;
        iss >> i0 >> i1 >> i2;

        int texCoordCount;
        iss >> texCoordCount;

        std::vector<float> tex;
        for (int j = 0; j < texCoordCount; ++j) {
            float val;
            iss >> val;
            tex.push_back(val);
        }

        // Chaque triangle → 3 sommets (non partagés)
        glm::ivec3 ids = { i0, i1, i2 };

        for (int j = 0; j < 3; ++j) {
            glm::vec3 pos = tempPositions[ids[j]];
            glm::vec3 norm = tempNormals[ids[j]];
            float u = tex[j * 2];
            float v = tex[j * 2 + 1];

            vertices.push_back(pos.x);
            vertices.push_back(pos.y);
            vertices.push_back(pos.z);

            normals.push_back(norm.x);
            normals.push_back(norm.y);
            normals.push_back(norm.z);

            texCoords.push_back(u);
            texCoords.push_back(v);

            indices.push_back(indices.size());
        }
    }

    return true;
}

// Fonction pour charger une texture
unsigned int loadTexture(const char* path)

{
    unsigned int textureID2;
    glGenTextures(1, &textureID2);

    int width, height, nrChannels;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(path, &width, &height, &nrChannels, 0);

    if (data) {
        GLenum format = (nrChannels == 4) ? GL_RGBA : GL_RGB;
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        stbi_image_free(data);  // Libérer la mémoire
        std::cout << "Texture chargée avec succès: " << path << std::endl;
    }
    else {
        std::cerr << "Échec du chargement de la texture: " << path << std::endl;
        std::cerr << "Erreur STBI: " << stbi_failure_reason() << std::endl;
    }
    return textureID;
}
// Modification de la fonction processInput pour utiliser la vue de caméra actuelle
float overlayScale = 1.f;
// Fonction pour initialiser le plan d'overlay
void setupOverlay() {
    // Définir les sommets du plan (quad)
    float overlayVertices[] = {
        //  Positions         |  UVs inversés (v -> 1 - v)
        -overlayScale,  overlayScale, 0.0f, 0.0f, 1.0f,
         overlayScale,  overlayScale, 0.0f, 1.0f, 1.0f,
         overlayScale, -overlayScale, 0.0f, 1.0f, 0.0f,
        -overlayScale, -overlayScale, 0.0f, 0.0f, 0.0f
    };


    unsigned int overlayIndices[] = {
        0, 1, 2,
        0, 2, 3
    };

    // Créer le VAO, VBO et EBO pour l'overlay
    glGenVertexArrays(1, &overlayVAO);
    glGenBuffers(1, &overlayVBO);
    glGenBuffers(1, &overlayEBO);

    glBindVertexArray(overlayVAO);

    glBindBuffer(GL_ARRAY_BUFFER, overlayVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(overlayVertices), overlayVertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, overlayEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(overlayIndices), overlayIndices, GL_STATIC_DRAW);

    // Attributs de position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Attributs de coordonnées de texture
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    // Charger la texture de l'overlay (remplacez par le chemin de votre image)
    overlayTexture = loadTexture("D:/project/verifCam/shoe1.png"); // Remplacez par le chemin de votre image

    // Shaders pour l'overlay
    const char* overlayVertexShaderSource = R"(
         #version 330 core
         layout (location = 0) in vec3 aPos;
         layout (location = 1) in vec2 aTexCoord;
         
         out vec2 TexCoord;
         
         void main() {
             gl_Position = vec4(aPos, 1.0);
             TexCoord = aTexCoord;
         }
     )";

    const char* overlayFragmentShaderSource = R"(
         #version 330 core
         in vec2 TexCoord;
         out vec4 FragColor;
         
         uniform sampler2D overlayTexture;
         uniform float transparency = 0.7; // Contrôle la transparence
         
         void main() {
             vec4 texColor = texture(overlayTexture, TexCoord);
             // Appliquer la transparence globale
                FragColor = vec4(texColor.rgb, texColor.a * transparency);
              
         }
     )";

    overlayShaderProgram = createShaderProgram(overlayVertexShaderSource, overlayFragmentShaderSource);
}

// Convertir la matrice de pose en matrices de vue et de modèle OpenGL
glm::mat4 convertPoseToViewMatrix(const std::vector<std::vector<double>>& pose) {
    // La matrice de pose est généralement [R|t] où R est une matrice de rotation 3x3 et t est un vecteur de translation
    glm::mat4 viewMatrix(1.0f); // Matrice identité par défaut

    // Remplir la matrice à partir des données de pose
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i < pose.size() && j < pose[i].size()) {
                viewMatrix[i][j] = static_cast<float>(pose[i][j]); // Notez l'inversion des indices i et j
            }
        }
    }
    viewMatrix[1] = -viewMatrix[1];
    viewMatrix[2] = -viewMatrix[2];
    return viewMatrix;
}


// Convertir les paramètres intrinsèques en matrice de projection OpenGL
glm::mat4 createProjectionMatrixFromIntrinsics(const CameraParams& params, float nearPlane, float farPlane) {
    // Créer une matrice de projection à partir des paramètres intrinsèques
    glm::mat4 projection = glm::mat4(0.0f);

    // Remplir la matrice de projection
    projection[0][0] = static_cast<float>(2 * params.focal_length_x / params.image_width);
    projection[1][1] = static_cast<float>(2 * params.focal_length_y / params.image_height);

    // Décalage du centre de l'image
    projection[2][0] = static_cast<float>((2 * params.principal_point_x / params.image_width) - 1.0);
    projection[2][1] = static_cast<float>((2 * params.principal_point_y / params.image_height) - 1.0);

    // Paramètres de profondeur
    projection[2][2] = static_cast<float>( -((farPlane + nearPlane) / (farPlane - nearPlane)));
    projection[3][2] = static_cast<float>(-(2 * farPlane * nearPlane) / (farPlane - nearPlane));

    projection[2][3] = -1.0f;

    return projection;
}




// Fonction pour configurer la caméra dans OpenGL
void setupCameraForOpenGL(const json& cameraData) {
    try {
        // Extraire les paramètres de la caméra
        CameraParams params;
        params.focal_length_x = cameraData["camera_params"]["focal_length_x"];
        params.focal_length_y = cameraData["camera_params"]["focal_length_y"];
        params.principal_point_x = cameraData["camera_params"]["principal_point_x"];
        params.principal_point_y = cameraData["camera_params"]["principal_point_y"];

        // Ces valeurs sont généralement les dimensions de l'image pour laquelle les paramètres ont été calibrés
        // Si elles ne sont pas explicitement fournies, nous pouvons les définir à partir des points principaux
        params.image_width = 2 * static_cast<int>(params.principal_point_x);
        params.image_height = 2 * static_cast<int>(params.principal_point_y);

        // Extraire la matrice de pose
        std::vector<std::vector<double>> pose;
        for (const auto& row : cameraData["pose"]) {
            std::vector<double> poseRow;
            for (const auto& val : row) {
                poseRow.push_back(val);
            }
            pose.push_back(poseRow);
        }

        // Définir les plans de coupe
        float nearPlane = 0.001f;
        float farPlane = 2.0f;
        // Créer les matrices de vue et de projection
         viewMatrix = convertPoseToViewMatrix(pose);
         projectionMatrix = createProjectionMatrixFromIntrinsics(params, nearPlane, farPlane);
         viewMatrix = transpose(viewMatrix);
    }
    catch (const std::exception& e) {
        std::cerr << "Erreur lors de la configuration de la caméra: " << e.what() << std::endl;
    }
}


//fonction pour lire le json de la position 
 void readJsonFile()
{
    Position position;
	std::ifstream fichier("D:/project/verifCam/camera.json");
    if (!fichier.is_open()) {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier JSON" << std::endl;
        return;
    }

        json jsonData;
        fichier >> jsonData;
        fichier.close();
        // 3. Parcourir les caméras
        for (size_t i = 0; i < jsonData.size(); ++i) {
            const auto& camera = jsonData[i];
            setupCameraForOpenGL(jsonData[i]);
            // Extraire les informations
            std::string name = camera["name"];

            // Extraire la position depuis le nom
                 position = extractPositionFromName(name);
        }
}

int main() {
    // Initialisation de GLFW
    if (!glfwInit()) {
        std::cerr << "Échec de l'initialisation de GLFW" << std::endl;
        return -1;
    }

    // Configuration de la version OpenGL
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Création de la fenêtre
    GLFWwindow* window = glfwCreateWindow(1000, 1000, "Fenêtre OpenGL", nullptr, nullptr);
    if (!window) {
        std::cerr << "Échec de la création de la fenêtre GLFW" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Initialisation de GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Échec de l'initialisation de GLEW" << std::endl;
        return -1;
    }
    // Définir la fonction de callback pour le redimensionnement
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    // Activer le blending pour la transparence
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // Activer le test de profondeur
    glEnable(GL_DEPTH_TEST);

	setupOverlay();
    if (!loadPLY_clean("D:/project/PComNorm/Chemi-AU-O0051.ply", vertices, indices, normals, texCoords, textureFiles)) {
        return -1;
    }

    //------------------------------------------------
    // Création du VAO, VBO et EBO avec support pour les normales
    GLuint VAO, VBO, normalVBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &normalVBO);
    glGenBuffers(1, &EBO);
    glGenBuffers(1, &texCoordVBO);
    glBindVertexArray(VAO);

    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Paramètres de texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);


    // Vérifier si nous avons des chemins de texture
    if (!textureFiles.empty()) {
        // Utiliser le premier fichier de texture trouvé dans le PLY
        std::string texturePath = "D:/project/PComNorm/" + textureFiles[0];  // Ajuster le chemin si nécessaire

        int texWidth, texHeight, texChannels;
        unsigned char* data = stbi_load(texturePath.c_str(), &texWidth, &texHeight, &texChannels, 0);
        if (data) {
            GLenum format = (texChannels == 4) ? GL_RGBA : GL_RGB;
            glTexImage2D(GL_TEXTURE_2D, 0, format, texWidth, texHeight, 0, format, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D);
            stbi_image_free(data);  // Libérer la mémoire
            std::cout << "Texture chargée avec succès: " << texturePath << std::endl;
        }
        else {
            std::cerr << "Échec du chargement de la texture: " << texturePath << std::endl;
            std::cerr << "Erreur STBI: " << stbi_failure_reason() << std::endl;
        }
    }
    else {
        std::cerr << "Aucun fichier de texture trouvé dans le PLY" << std::endl;
    }

    // Buffer pour les positions
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Buffer pour les normales
    glBindBuffer(GL_ARRAY_BUFFER, normalVBO);
    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), normals.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);

    // Buffer pour les indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    // Buffer pour les texCoords
    glBindBuffer(GL_ARRAY_BUFFER, texCoordVBO);
    glBufferData(GL_ARRAY_BUFFER, texCoords.size() * sizeof(float), texCoords.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(2);



    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Shaders pour le modèle 3D avec illumination
    const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal; // Ajout des normales
    layout (location = 2) in vec2 aTexCoord;
    uniform mat4 Model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform mat4 MVP;
    
    out vec3 FragPos;
    out vec3 Normal;
    out vec2 TexCoord;
    void main() {
        gl_Position = MVP * vec4(aPos, 1.0);
        FragPos = vec3(Model * vec4(aPos, 1.0));
        // Transposer l'inverse de la matrice modèle pour les normales
        Normal = mat3(transpose(inverse(Model))) * aNormal;
        TexCoord = aTexCoord;
    }
)";

    const char* fragmentShaderSource = R"(
    #version 330 core
    in vec3 FragPos;
    in vec3 Normal;
    in vec2 TexCoord;
    out vec4 FragColor;
    
    // Paramètres d'éclairage
    uniform vec3 lightPos;
    uniform vec3 viewPos;
    uniform vec3 lightColor;

    uniform float ambientStrength;
    uniform float specularStrength;
    uniform float shininess;
    uniform sampler2D texture1;
    void main() {
               vec3 texColor = texture(texture1, TexCoord).rgb;
               vec3 ambient = ambientStrength * lightColor * texColor;
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor * texColor;
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
            vec3 specular = specularStrength * spec * lightColor;

            vec3 result = ambient + diffuse + specular;
           // FragColor = vec4(result, 1.0);
            FragColor = vec4(texture(texture1, TexCoord).rgb, 1.0);
    }
)";

    shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);

    // Paramètres d'éclairage par défaut
    glm::vec3 lightPos = glm::vec3(2.0f, 2.0f, 2.0f);
    glm::vec3 lightColor = glm::vec3(1.0f, 1.0f, 1.0f);
    glm::vec3 objectColor = glm::vec3(0.7f, 0.7f, 0.7f);
    float ambientStrength = 0.3f;
    float specularStrength = 0.5f;
    float shininess = 32.0f;

    bool succ = false;
	cout << "matrice projection  initial"<< endl;
    for (int i = 0; i < 4; i++) {
        std::cout << "  ";
        for (int j = 0; j < 4; j++) {
            std::cout << projection[i][j] << " ";
        }
        std::cout << std::endl;
    }
   

    // Boucle de rendu
    readJsonFile();
   
    view = viewMatrix; 
    /*
   glm::mat4  secondview =glm::lookAt(cameraPos,glm::vec3(0.0,0.03,-0.001), glm::vec3(0.0f, 1.0f, 0.0f));
   cout << "mlook at" << endl;
   for (int i = 0; i < 4; i++) {
       std::cout << "  ";
       for (int j = 0; j < 4; j++) {
           std::cout << secondview[i][j] << " ";
       }
       std::cout << std::endl;
   }*/
	projection = projectionMatrix;  
    while (!glfwWindowShouldClose(window)) {
        // Effacer les buffers de couleur et de profondeur
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // Matrice Model pour l'objet 3D
        Model = glm::mat4(1.0f);
        MVP = projection * view * Model;

        // Dessiner d'abord l'objet 3D
        glUseProgram(shaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "MVP"), 1, GL_FALSE, &MVP[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "Model"), 1, GL_FALSE, &Model[0][0]);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glUniform1i(glGetUniformLocation(shaderProgram, "texture1"), 0);

        // Passer les paramètres d'éclairage au shader
        glUniform3fv(glGetUniformLocation(shaderProgram, "lightPos"), 1, glm::value_ptr(lightPos));
        glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, glm::value_ptr(cameraPos));
        glUniform3fv(glGetUniformLocation(shaderProgram, "lightColor"), 1, glm::value_ptr(lightColor));
        glUniform3fv(glGetUniformLocation(shaderProgram, "objectColor"), 1, glm::value_ptr(objectColor));
        glUniform1f(glGetUniformLocation(shaderProgram, "ambientStrength"), ambientStrength);
        glUniform1f(glGetUniformLocation(shaderProgram, "specularStrength"), specularStrength);
        glUniform1f(glGetUniformLocation(shaderProgram, "shininess"), shininess);

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        // Échange des buffers
        
        glDisable(GL_DEPTH_TEST);
        
       
        // Dessiner le plan d'overlay
        glUseProgram(overlayShaderProgram);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, overlayTexture);
        glUniform1i(glGetUniformLocation(overlayShaderProgram, "overlayTexture"), 0);
        glBindVertexArray(overlayVAO);
        glDrawElements(GL_TRIANGLES,6, GL_UNSIGNED_INT, 0);

        // Réactiver le test de profondeur
        glEnable(GL_DEPTH_TEST);
        
        glfwSwapBuffers(window);
        glfwPollEvents();

    }
    std::cout << "Matrice de vue:" << std::endl;
    for (int i = 0; i < 4; i++) {
        std::cout << "  ";
        for (int j = 0; j < 4; j++) {
            std::cout << view[i][j] << " ";
        }
        std::cout << std::endl;
    }


    // Nettoyage
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteProgram(shaderProgram);
    glDeleteBuffers(1, &normalVBO);
    glDeleteBuffers(1, &texCoordVBO);
    glDeleteProgram(overlayShaderProgram);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
