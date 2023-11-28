output_file="data.zip"
gdown "https://drive.google.com/uc?id=1uWaavrpKKllVSQ73TTuLWPc4aqVvrkpx" -O "$output_file"
echo "Download complete. File saved as $output_file."


unzip "$output_file" -d /home/afrin/cs635_KGQA/EmbedKGQA/

rm "$output_file" 