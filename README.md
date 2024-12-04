# QueryPlot - Generating mineral evidence layers from geological queries
This tool is developed by SRI TA2 team for USGS under DARPA CriticalMAAS program. QueryPlot uses transformer encoder to extract sentence embeddings from polygon descriptions and compare them with a user query embedding to find the top relevant polygons. A user can use either custom query or pre-defined descriptive deposit models to search polygons and download the result layers for further processing in local GIS software.

## System setup (on AWS)
1. Create an EC2 instance and connect to it through console
    
    -   System spec (recommend m8g.2xlarge equivalent or higher)
        -   CPU: 2.3 GHz
        -   Number of (v)CPUs: 8
        -   Memory: 32 GiB
        -   Storage: 200 GiB
        -   GPU: not required
    -   Security Groups setting
        -   add type `Custom TCP` on port `8501` to `Inbound rules`

2. Simply run the setup script
    ```bash
    bash setup_aws.sh
    ```
    what this script does:

    1. Install dependencies
        ```bash
        sudo apt update
        sudo apt install -y python3-pip
        sudo apt install -y python3.12-venv
        sudo apt-get install -y libgtk-3-dev
        sudo apt-get install -y libgdal-dev gdal-bin python3-gdal
        sudo apt-get install -y python3.12-dev
        sudo apt-get install unzip
        export CPLUS_INCLUDE_PATH=/usr/include/gdal
        export C_INCLUDE_PATH=/usr/include/gdal
        ```

    2. Prepare python environment
        ```bash
        mkdir app
        cd app
        git clone https://github.com/DARPA-CRITICALMAAS/sri-ta2-mappable-criteria.git
        cd sri-ta2-mappable-criteria/
        python3 -m venv /home/ubuntu/venvs/sri-map-synth
        source /home/ubuntu/venvs/sri-map-synth/bin/activate
        pip install GDAL==`gdal-config --version`
        pip install -r polygon_ranking/requirements.txt
        ```

    3. Start server
        ```bash
        source /home/ubuntu/venvs/sri-map-synth/bin/activate
        nohup streamlit run Welcome.py > streamlit.log 2>&1 & 
        echo $! > streamlit_pid.txt
        ```
        *Note:*
        -   *QueryPlot is built with streamlit and it runs on port `8501` by default*
        -   *The logs will be written to file `streamlit.log`*
        -   *The PID will be stored in `streamlit_pid.txt` for terminating the process in the future*
    
3. Access QueryPlot in a browser

    To access it, open a browser and type in the ip address of your EC2 instance and the port number (e.g., `http://your.ec2.ip.address:8501`)

## Prepare polygon data
Once the server has been set up and the service is running, download the preprocessed SGMC shape file into corresponding directory:
```bash
cd /path/to/workdir-data/preproc/sgmc/
wget https://cmaas-ta2-sri-bucket.s3.us-east-2.amazonaws.com/SGMC_preproc_default.gpkg
```
Alternatively, you could also create the shapefile from scratch within QueryPlot by following the `How to create your own shapefile` section in the [user manual](https://docs.google.com/document/d/1WTDQBVn73pqW3YsGDRtNmBFUjEyRdCFV)

## Generate evidence map layers
Follow the instructions in [user manual](https://docs.google.com/document/d/1WTDQBVn73pqW3YsGDRtNmBFUjEyRdCFV) or watch this 8-minute [video](https://drive.google.com/file/d/1eSYXvgU6Voj8XXoXC2xKEyTE8t9aZun6) to learn how to use QueryPlot.