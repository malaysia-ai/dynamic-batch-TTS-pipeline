import base64
import io
import soundfile as sf
import tempfile
from fastapi.responses import FileResponse

def return_type(file_response, response_format, r):
    if file_response:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file_path = temp_file.name
            sf.write(temp_file_path, r['audio'], r['sr'], format=response_format.upper())
        return FileResponse(
            temp_file_path, 
            media_type='audio/mpeg', 
            filename=f'speech.{response_format}',
        )
    else:
        buffer = io.BytesIO()
        sf.write(buffer, r['audio'], samplerate= r['sr'], format=response_format.upper())
        buffer.seek(0)
        audio_binary = buffer.read()
        audio_base64 = base64.b64encode(audio_binary).decode('utf-8')
        return {
            'audio': audio_base64,
            'stats': r['stats']
        }