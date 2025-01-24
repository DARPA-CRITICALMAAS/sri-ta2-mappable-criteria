
<p align="center">
  <img width="300" src="images/logo.png">
</p>

# QueryPlot - Generating mineral evidence maps from geological queries
This tool is developed by SRI TA2 team for USGS under DARPA CriticalMAAS program. QueryPlot uses transformer encoder to extract sentence embeddings from polygon descriptions and compare them with a user query embedding to find the top relevant polygons. A user can use either custom query or pre-defined descriptive deposit models to search polygons and download the result layers for further processing in local GIS software.


## Installation (on AWS)
### Launch EC2 instance
-   Amazon Machine Image (AMI)
    -   Ubuntu Server 24.04 LTS
-   System spec (recommend m7a.2xlarge equivalent or higher)
    -   CPU: 3.7 GHz (**x86_64**)
    -   Cores: 8
    -   Arch: x86_64
    -   Number of (v)CPUs: 8
    -   Memory: 32 GiB
    -   Storage: 200 GiB
    -   GPU: not required
-   Security Group - inbound rules setting
    -   [Using reverse proxy ([instructions](#nginx-and-ssl-certificate))] (recommended): Open port `80` (HTTP), `443` (HTTPS), and `22` (SSH). The EC2 default security group will open these ports by default.
    -   [Opening port directly] (not recommended): Add type `Custom TCP` on port `8501`.

### Deployment using pre-built image from DockerHub
Simply run the script
```bash
QUERYPLOT_PWD=<Create a password> CDR_KEY=<Your CDR key> bash setup_docker.sh
```
This script automatically installs docker (will skip if docker is already installed), pulls the pre-built image, download necessary data artifacts, and runs the container.
<details>
  <summary>More details</summary>

1.  Install Docker
    ```bash
    # https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository
    # Add Docker's official GPG key:
    sudo apt-get update
    sudo apt-get install ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # Add the repository to Apt sources:
    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update

    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    # Add current user to the Docker group (optional, avoids using 'sudo' for Docker commands)
    sudo usermod -aG docker $USER
    ```

2.  Pull image from DockerHub
    ```bash
    docker pull mye1225/cmaas-sri-queryplot:latest
    ```

    Alternatively, you can also build docker image from source
    ```bash
    git clone https://github.com/DARPA-CRITICALMAAS/sri-ta2-mappable-criteria.git
    cd sri-ta2-mappable-criteria
    docker build -t cmaas-sri-queryplot .
    ```

3. Download data artifacts
    ```bash
    echo "Downloading data artifacts"
    echo "[Shapefile] SGMC_preproc_default.gpkg ..."
    mkdir -p $HOME/app/workdir-data/preproc/sgmc
    cd $HOME/app/workdir-data/preproc/sgmc/
    wget https://cmaas-ta2-sri-bucket.s3.us-east-2.amazonaws.com/SGMC_preproc_default.gpkg

    echo "[Deposit model] _Default_.json, GPT_generated.json ..."
    mkdir -p $HOME/app/workdir-data/deposit_models
    cd $HOME/app/workdir-data/deposit_models/
    wget https://cmaas-ta2-sri-bucket.s3.us-east-2.amazonaws.com/_Default_.json
    wget https://cmaas-ta2-sri-bucket.s3.us-east-2.amazonaws.com/GPT_generated.json
    ```

4. Create `secrets.toml` file
    ```bash
    # export QUERYPLOT_PWD=<Your password>
    # export CDR_KEY=<Your CDR key>
    # export OPENAI_KEY=<Your OpenAI key>  

    # create secret file from envs
    echo "Creating 'secrets.toml'"
    cat > $HOME/app/secrets.toml << EOF
    password = "${QUERYPLOT_PWD:-CriticalMaas}"
    cdr_key = "${CDR_KEY:-}"
    openai_key = "${OPENAI_KEY:-}"
    EOF
    ```

5. Create and run container
    ```bash
    docker run \
    -d \
    -v $HOME/app/secrets.toml:/home/ubuntu/app/sri-ta2-mappable-criteria/.streamlit/secrets.toml \
    -v $HOME/app/workdir-data:/home/ubuntu/app/workdir-data \
    -p 8501:8501 \
    mye1225/cmaas-sri-queryplot:latest
    ```
    *Note: You could optionally create a new `config.toml` file (e.g., `$HOME/app/config.toml`) to replace the default one:*
    ```toml
    [vars]
    sys_ver = 'v1.3'
    data_dir = '../workdir-data'

    [endpoints]
    cdr_cmas = 'https://api.cdr.land/v1/prospectivity/cmas'
    cdr_push = 'https://api.cdr.land/v1/prospectivity/datasource'

    [params]
    percentile_threshold_min = 80
    percentile_threshold_default = 90
    raster_height = 500
    raster_width = 500
    map_base = 'Cartodb Positron'
    map_polygon_opacity = 0.8
    ```
    *If you do that, make sure to mount it to the container for the modification to be effective:*
    ```bash
    docker run \
    -d \
    -v $HOME/app/secrets.toml:/home/ubuntu/app/sri-ta2-mappable-criteria/.streamlit/secrets.toml \
    -v $HOME/app/config.toml:/home/ubuntu/app/sri-ta2-mappable-criteria/config.toml
    -v $HOME/app/workdir-data:/home/ubuntu/app/workdir-data \
    -p 8501:8501 \
    mye1225/cmaas-sri-queryplot:latest
    ```
</details>

### Nginx and SSL certificate
1. Install Nginx (if it has not been installed):
```bash
sudo apt install nginx -y
```

2. Create a new Nginx config file for QueryPlot:
```
sudo vim /etc/nginx/sites-available/queryplot
```

3. Add the follwing configuration:
```nginx
server {
    listen 443 ssl;
    server_name <your.domain.name>;

    ssl_certificate </path/to/cert.pem>;
    ssl_certificate_key </path/to/key.pem>;

    location / {
        proxy_pass http://0.0.0.0:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
    }

    client_max_body_size 500M;
}
```
*Note: Please make sure nginx is able to access the SSL certificate files.*

4. Create a symbolic link to enable this configuration:
```bash
sudo ln -s /etc/nginx/sites-available/queryplot /etc/nginx/sites-enabled/
```

5. Test Nginx configuration and restart
```bash
sudo nginx -t
sudo systemctl restart nginx
```

### Building docker image locally
Run this simple docker build command
```bash
docker build -t cmaas-sri-queryplot .
```

### Run python script manually
1.  Pull code
    ```bash
    mkdir $HOME/app
    cd $HOME/app
    git clone https://github.com/DARPA-CRITICALMAAS/sri-ta2-mappable-criteria.git
    cd sri-ta2-mappable-criteria/
    ```

2. Setup environment
    ```bash
    bash setup_aws.sh
    ```
    This script will do the following things:

    ```bash
    # Install dependencies
    sudo apt update
    sudo apt install -y python3-pip
    sudo apt install -y python3.12-venv
    sudo apt-get install -y libgtk-3-dev
    sudo apt-get install -y libgdal-dev gdal-bin python3-gdal
    sudo apt-get install -y python3.12-dev
    sudo apt-get install unzip
    export CPLUS_INCLUDE_PATH=/usr/include/gdal
    export C_INCLUDE_PATH=/usr/include/gdal

    # Prepare python environment
    python3 -m venv /home/ubuntu/venvs/sri-map-synth
    source /home/ubuntu/venvs/sri-map-synth/bin/activate
    pip install GDAL==`gdal-config --version`
    pip install -r polygon_ranking/requirements.txt
    ```

3.  Setup USGS DOI SSL

4.  Download data artifacts
    ```bash
    echo "[Shapefile] SGMC_preproc_default.gpkg ..."
    mkdir -p $HOME/app/workdir-data/preproc/sgmc
    cd $HOME/app/workdir-data/preproc/sgmc/
    wget https://cmaas-ta2-sri-bucket.s3.us-east-2.amazonaws.com/SGMC_preproc_default.gpkg

    echo "[Deposit model] _Default_.json, GPT_generated.json ..."
    mkdir -p $HOME/app/workdir-data/deposit_models
    cd $HOME/app/workdir-data/deposit_models/
    wget https://cmaas-ta2-sri-bucket.s3.us-east-2.amazonaws.com/_Default_.json
    wget https://cmaas-ta2-sri-bucket.s3.us-east-2.amazonaws.com/GPT_generated.json
    ```

5.  Secrets
    ```bash
    cd $HOME/app/sri-ta2-mappable-criteria/
    cat > .streamlit/secrets.toml << EOF
    password = "<Create a password>"
    cdr_key = "<Your CDR Key>"
    openai_key = "<Your OpenAI Key>"  # This is optional
    EOF
    ```

6.  Start service
    ```bash
    bash start_server.sh
    ```
    *Note:*
    -   *QueryPlot is built with streamlit and it runs on port `8501` by default*
    -   *The logs will be written to file `streamlit.log`*
    -   *The PID will be stored in `streamlit_pid.txt` for terminating the process in the future*
    
### Use the tool
To access QueryPlot, open a browser and type in the IP address of your EC2 instance and the port number (e.g., `http://your.ec2.ip.address:8501`)

Follow the instructions in [user manual](https://docs.google.com/document/d/1WTDQBVn73pqW3YsGDRtNmBFUjEyRdCFV) or watch this 8-minute [video](https://drive.google.com/file/d/1eSYXvgU6Voj8XXoXC2xKEyTE8t9aZun6) to learn how to use QueryPlot.

### License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
