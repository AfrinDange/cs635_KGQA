output_file="dataset/IMDB.csv"
gdown "https://drive.google.com/uc?id=1lKeP9E_S8uuaMu12lv8snzVKXXHG99ck" -O "$output_file"
echo "Download complete. File saved as $output_file."

output_file="dataset/QA.txt"
gdown "https://drive.google.com/uc?id=1m84oGmiCP06CBwiLgUTkuV9708-xcLQ0" -O "$output_file"
echo "Download complete. File saved as $output_file."

output_file="dataset/test.txt"
gdown "https://drive.google.com/uc?id=1YQhw_utiHigVsyv6s7sUlmFO2oaiyztX" -O "$output_file"
echo "Download complete. File saved as $output_file."
