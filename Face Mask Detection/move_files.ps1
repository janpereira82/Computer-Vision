# Criar pasta temporária
New-Item -ItemType Directory -Path "temp_face_mask"

# Mover todos os arquivos e pastas (exceto .git e temp_face_mask) para a pasta temporária
Get-ChildItem -Path "." -Exclude @(".git", "temp_face_mask", "move_files.ps1") | Move-Item -Destination "temp_face_mask"

# Criar a pasta Face Mask Detection
New-Item -ItemType Directory -Path "Face Mask Detection"

# Mover o conteúdo da pasta temporária para Face Mask Detection
Get-ChildItem -Path "temp_face_mask" | Move-Item -Destination "Face Mask Detection"

# Remover a pasta temporária
Remove-Item -Path "temp_face_mask"
