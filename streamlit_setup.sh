mkdir -p ~/.streamlit/

echo "[general]
email = \"stepheechrome@gmail.com\"
" > ~/.streamlit/credentials.toml

echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml

wget https://github.com/stephenllh/mnist-mlops/releases/latest/download/mnist_cnn.pt