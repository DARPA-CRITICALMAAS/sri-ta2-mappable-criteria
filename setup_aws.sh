sudo apt update
sudo apt install -y python3-pip
sudo apt install -y python3.12-venv
sudo apt-get install -y libgtk-3-dev
sudo apt-get install -y libgdal-dev gdal-bin python3-gdal
sudo apt-get install -y python3.12-dev
sudo apt-get install unzip
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal

# # >>> start USGS DOI SSL
# cp DOIRootCA2.crt /usr/local/share/ca-certificates
# chmod 644 /usr/local/share/ca-certificates/DOIRootCA2.crt && \
#     update-ca-certificates
# # you probably don't need all of these, but they don't hurt 
# export PIP_CERT="/etc/ssl/certs/ca-certificates.crt" \
#     SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt" \
#     CURL_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt" \
#     REQUESTS_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt" \
#     AWS_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
# # <<< end USGS DOI SSL

# mkdir app
# cd app
# git clone https://github.com/DARPA-CRITICALMAAS/sri-ta2-mappable-criteria.git
# cd sri-ta2-mappable-criteria/
python3 -m venv $HOME/venvs/sri-map-synth
source $HOME/venvs/sri-map-synth/bin/activate
pip install GDAL==`gdal-config --version`
pip install -r polygon_ranking/requirements.txt

# bash start_server.sh