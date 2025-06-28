# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(
    ['mcp_speech_to_text_server/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('mcp_tool_env/lib/python3.11/site-packages/whisper/assets', 'whisper/assets')
    ],
    hiddenimports=['sounddevice._sounddevice'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='mcp-speech-to-text-server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='mcp-speech-to-text-server'
) 