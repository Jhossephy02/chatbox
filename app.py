# IMPORTAR LAS LIBRERIAS
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from flask_cors import CORS
from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel
from gtts import gTTS
import requests
import json
import re
from datetime import datetime
import os

# --- CONFIGURACIÓN ---

# Directorios y archivos
UPLOAD_FOLDER = 'uploads'  # donde se guardan los audios subidos
TTS_FOLDER = 'tts'  # donde se guardan los audios generados por TTS
CHAT_FILE = 'chat.json'  # archivo para historial de chat
CLINIC_INFO_FILE = 'clinic_info.txt'  # archivo con info de la clínica

# CAMBIO REQ 4: Configuración de ElevenLabs
# !! IMPORTANTE: Reemplaza con tu clave de API real de ElevenLabs !!
ELEVENLABS_API_KEY = 'TU_API_KEY_DE_ELEVENLABS_AQUI' 
# Puedes encontrar IDs de voz aquí: https://api.elevenlabs.io/v1/voices
VOICE_ID = '21m00Tcm4TlvDq8ikWAM'  # Voz predeterminada (Adam)
ELEVEN_MODEL = 'eleven_multilingual_v2'  # Modelo actualizado para español

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TTS_FOLDER, exist_ok=True)

# Cargar modelo Whisper (versión ligera)
try:
    model = WhisperModel("base", device="cpu", compute_type="int8")
    print("Modelo Whisper 'base' cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar modelo Whisper: {e}")
    # Intenta con un modelo más pequeño si falla
    try:
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("Modelo Whisper 'tiny' cargado como fallback.")
    except Exception as e2:
        print(f"Error crítico: No se pudo cargar ningún modelo Whisper: {e2}")
        exit() # Salir si no se puede cargar whisper

# Crear la app Flask y habilitar CORS
app = Flask(__name__)
CORS(app)

# --- FUNCIONES DE AYUDA ---

# CARGAR LA INFORMACION DE LA CLINICA (No se usa activamente pero está disponible)
def cargar_info_clinica():
    try:
        with open(CLINIC_INFO_FILE, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception:
        return "No se pudo cargar la información de la clínica."

# GUARDAR LOS CHAT GENERALES EN UN ARCHIVO JSON
def guardar_chat_historial(usuario_text, asistente_text):
    entry = {
        'fecha': datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        'usuario': usuario_text,
        'asistente': asistente_text,
    }
    try:
        if not os.path.exists(CHAT_FILE):
            data = []
        else:
            with open(CHAT_FILE, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = []
                except json.JSONDecodeError:
                    data = []
        
        data.append(entry)
        
        with open(CHAT_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Error al guardar el historial de chat: {e}")

# Conexión con Ollama
def responder_con_ollana(prompt, model_name="phi3"):
    # Cargar contexto de la clínica (si existe)
    # info_clinica = cargar_info_clinica()
    # prompt_contextual = f"Responde la siguiente pregunta basándote en esta información:\n---{info_clinica}---\n\nPregunta: {prompt}\n\nSi la respuesta no está en la información, responde normalmente."
    
    # Prompt simple (como estaba en el original)
    prompt_final = f"Por favor responde en español de forma concisa y amigable.\nUsuario: {prompt}\nAsistente:"
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt_final, "max_tokens": 512, "stream": False}, # Stream=False para respuesta completa
            timeout=60
        )
        
        response.raise_for_status() # Lanza error si la respuesta no es 200
        
        # Con stream=False, la respuesta es un JSON completo
        data = response.json()
        full_text = data.get('response', '').strip()
        
        return full_text or "No se obtuvo respuesta del modelo."
        
    except requests.exceptions.ConnectionError:
        return "Error: No se pudo conectar con Ollama. Asegúrate de que esté en ejecución."
    except requests.exceptions.Timeout:
        return "Error: La solicitud a Ollama tardó demasiado."
    except Exception as e:
        return f"Error al conectar con Ollama: {e}"

# --- CAMBIO REQ 4: FUNCIONES DE TEXT-TO-SPEECH (TTS) ---

# 1. Función para ElevenLabs (Prioridad 1)
def generar_audio_tts_elevenlabs(texto):
    # Si la API key es el placeholder o está vacía, no intentes llamar a la API
    if not ELEVENLABS_API_KEY or ELEVENLABS_API_KEY == 'TU_API_KEY_DE_ELEVENLABS_AQUI':
        print("API Key de ElevenLabs no configurada. Omitiendo.")
        return None
    
    # Ruta de guardado (siempre el mismo archivo para sobreescribir)
    audio_path = os.path.join(TTS_FOLDER, "respuesta.mp3")
    
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "text": texto,
            "model_id": ELEVEN_MODEL,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        
        print("Generando audio con ElevenLabs...")
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        
        if resp.status_code == 200:
            with open(audio_path, 'wb') as audio_file:
                audio_file.write(resp.content)
            print("Audio generado con ElevenLabs exitosamente.")
            return audio_path
        else:
            print(f"Error de ElevenLabs API: {resp.status_code} - {resp.text}")
            return None
            
    except Exception as e:
        print(f"Excepción al llamar a ElevenLabs: {e}")
        return None

# 2. Función para gTTS (Fallback / Respaldo)
def generar_audio_tts_gtts(text, idioma="es"):
    # Ruta de guardado
    output_path = os.path.join(TTS_FOLDER, "respuesta.mp3")
    
    try:
        print("Generando audio con gTTS (fallback)...")
        tts = gTTS(text=text, lang=idioma)
        tts.save(output_path)
        print("Audio generado con gTTS exitosamente.")
        return output_path
    except Exception as e:
        print(f"Error en gTTS: {e}")
        return None

# --- RUTAS DE LA APLICACIÓN FLASK ---

# Ruta principal para servir el HTML
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para procesar el audio grabado
@app.route('/upload', methods=['POST'])
def upload_audio():
    try:
        file = request.files['audio']
        filename = "grabacion.webm" # Nombre fijo
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Transcribir el audio (audio) a texto
        segments, info = model.transcribe(file_path, language="es") # Forzar español
        original_text = " ".join([segment.text for segment in segments]).strip()

        # Si se produce algun error no hay texto
        if not original_text:
            return jsonify({'error': 'No se detectó texto en el audio.'}), 400

        # Obtener la respuesta de Ollama (phi3)
        respuesta = responder_con_ollana(original_text, model_name="phi3")
        
        # Guardar en el historial
        guardar_chat_historial(original_text, respuesta)

        # CAMBIO REQ 4: Lógica de TTS (ElevenLabs primero, gTTS como fallback)
        tts_path = generar_audio_tts_elevenlabs(respuesta)
        if not tts_path:
            tts_path = generar_audio_tts_gtts(respuesta, idioma="es")

        return jsonify({
            "original": original_text,
            "asistente": respuesta,
            "tts_audio": '/tts/respuesta' if tts_path else None
        }), 200  # HTTP 200 EXITO

    except Exception as e:
        print(f"Error en /upload: {e}")
        return jsonify({'error': str(e)}), 500  # HTTP 500 ERROR DEL SERVIDOR

# CAMBIO REQ 1: Nueva ruta para procesar texto
@app.route('/chat-text', methods=['POST'])
def chat_text():
    try:
        data = request.get_json()
        original_text = data.get('text')
        
        if not original_text:
            return jsonify({'error': 'No se recibió texto.'}), 400

        # Obtener la respuesta de Ollama (phi3)
        respuesta = responder_con_ollana(original_text, model_name="phi3")
        
        # Guardar en el historial
        guardar_chat_historial(original_text, respuesta)

        # CAMBIO REQ 4: Lógica de TTS (ElevenLabs primero, gTTS como fallback)
        tts_path = generar_audio_tts_elevenlabs(respuesta)
        if not tts_path:
            tts_path = generar_audio_tts_gtts(respuesta, idioma="es")

        return jsonify({
            "original": original_text,
            "asistente": respuesta,
            "tts_audio": '/tts/respuesta' if tts_path else None
        }), 200  # HTTP 200 EXITO
        
    except Exception as e:
        print(f"Error en /chat-text: {e}")
        return jsonify({'error': str(e)}), 500

# Ruta para servir el archivo de audio generado
@app.route('/tts/respuesta')
def servir_tts():
    audio_path = os.path.join(TTS_FOLDER, "respuesta.mp3")
    if os.path.exists(audio_path):
        # Agregar 'no-cache' para evitar que el navegador guarde el audio antiguo
        response = send_file(audio_path, mimetype='audio/mpeg')
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    else:
        return jsonify({'error': 'Archivo de audio no encontrado.'}), 404

# EJECUTAR LA APLICACION
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)