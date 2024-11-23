import argparse
import logging
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description='Configuration parser')

    parser.add_argument(
        '--host', type=str, default=os.environ.get('HOSTNAME', '0.0.0.0'),
        help='host name to host the app (default: %(default)s, env: HOSTNAME)'
    )
    parser.add_argument(
        '--port', type=int, default=int(os.environ.get('PORT', '7088')),
        help='port to host the app (default: %(default)s, env: PORT)'
    )
    parser.add_argument(
        '--loglevel', default=os.environ.get('LOGLEVEL', 'INFO').upper(),
        help='Logging level (default: %(default)s, env: LOGLEVEL)'
    )
    parser.add_argument(
        '--reload', type=lambda x: x.lower() == 'true',
        default=os.environ.get('reload', 'false').lower() == 'true',
        help='Enable hot loading (default: %(default)s, env: RELOAD)'
    )
    parser.add_argument(
        '--enable-speech-enhancement', type=lambda x: x.lower() == 'true',
        default=os.environ.get('ENABLE_SPEECH_ENHANCEMENT', 'true').lower() == 'true',
        help='Enable document layout detection (default: %(default)s, env: ENABLE_SPEECH_ENHANCEMENT)'
    )
    parser.add_argument(
        '--model-speech-enhancement',
        default=os.environ.get('MODEL_SPEECH_ENHANCEMENT', 'resemble-enhance'),
        help='Model type (default: %(default)s, env: MODEL_SPEECH_ENHANCEMENT)'
    )
    parser.add_argument(
        '--enable-tts', type=lambda x: x.lower() == 'true',
        default=os.environ.get('ENABLE_TTS', 'true').lower() == 'true',
        help='Enable OCR (default: %(default)s, env: ENABLE_tts)'
    )
    parser.add_argument(
        '--model-tts',
        default=os.environ.get('MODEL_TTS', 'f5-tts'),
        help='Model type (default: %(default)s, env: MODEL_TTS)'
    )
    parser.add_argument(
        '--dynamic-batching-microsleep', type=float,
        default=float(os.environ.get('DYNAMIC_BATCHING_MICROSLEEP', '1e-4')),
        help='microsleep to group dynamic batching, 1 / 1e-4 = 10k steps for second (default: %(default)s, env: DYNAMIC_BATCHING_MICROSLEEP)'
    )
    parser.add_argument(
        '--dynamic-batching-speech-enhancement-batch-size', type=int,
        default=int(os.environ.get('DYNAMIC_BATCHING_SPEECH_ENHANCEMENT_BATCH_SIZE', '16')),
        help='maximum of batch size for speech enhancement during dynamic batching (default: %(default)s, env: DYNAMIC_BATCHING_SPEECH_ENHANCEMENT_BATCH_SIZE)'
    )
    parser.add_argument(
        '--dynamic-batching-ts-batch-size', type=int,
        default=int(os.environ.get('DYNAMIC_BATCHING_TTS_BATCH_SIZE', '16')),
        help='maximum of batch size for TTS during dynamic batching (default: %(default)s, env: DYNAMIC_BATCHING_TTS_BATCH_SIZE)'
    )
    parser.add_argument(
        '--accelerator-type', default=os.environ.get('ACCELERATOR_TYPE', 'cuda'),
        help='Accelerator type (default: %(default)s, env: ACCELERATOR_TYPE)'
    )
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=int(os.environ.get('MAX_CONCURRENT', '100')),
        help='Maximum concurrent requests (default: %(default)s, env: MAX_CONCURRENT)'
    )

    args = parser.parse_args()

    if args.model_speech_enhancement not in {'resemble-enhance'}:
        raise ValueError('Currently Speech Enhancement, `--model-speech-enhancement` or `MODEL_SPEECH_ENHANCEMENT` environment variable, only support https://github.com/resemble-ai/resemble-enhance')

    if args.model_tts not in {'f5-tts'}:
        raise ValueError('Currently TTS, `--model-tts` or `MODEL_TTS` environment variable, only support https://github.com/SWivid/F5-TTS')

    return args


args = parse_arguments()

logging.basicConfig(level=args.loglevel)

logging.info(f'Serving app using {args}')