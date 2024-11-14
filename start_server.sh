source /home/ubuntu/venvs/sri-map-synth/bin/activate
nohup streamlit run Welcome.py > streamlit.log 2>&1 & 
echo $! > streamlit_pid.txt