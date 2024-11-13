sudo apt update
sudo apt install -y python3-pip
sudo apt install -y python3.12-venv
sudo apt-get install -y libgtk-3-dev
sudo apt-get install -y libgdal-dev gdal-bin python3-gdal
sudo apt-get install -y python3.12-dev
sudo apt-get install unzip
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal

mkdir app
cd app
git clone https://github.com/DARPA-CRITICALMAAS/sri-ta2-mappable-criteria.git
cd sri-ta2-mappable-criteria/
python3 -m venv /home/ubuntu/venvs/sri-map-synth
source /home/ubuntu/venvs/sri-map-synth/bin/activate
pip install GDAL==`gdal-config --version`
pip install -r polygon_ranking/requirements.txt

nohup streamlit run Welcome.py > streamlit.log 2>&1 & 
echo $! > streamlit_pid.txt