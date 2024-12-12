#!/bin/bash

# Function to check if Docker is installed
check_docker_installed() {
    if ! command -v docker &> /dev/null; then
        return 1  # Docker is not installed
    else
        return 0  # Docker is installed
    fi
}

# Function to install Docker
install_docker() {
    echo "Installing Docker..."

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
    
    echo "Docker installation completed."
}

# Check if Docker is installed
if check_docker_installed; then
    echo "Docker is already installed."
else
    echo "Docker is not installed on this system."
    read -p "Do you want to install Docker? (Yes/No): " response
    case $response in
        [Yy]* )
            install_docker
            ;;
        [Nn]* )
            echo "Docker is required to proceed. Exiting script."
            exit 1
            ;;
        * )
            echo "Invalid response. Exiting script."
            exit 1
            ;;
    esac
fi

# pull docker
echo "Pulling image 'mye1225/cmaas-sri-queryplot:1.2'"
sudo docker pull mye1225/cmaas-sri-queryplot:1.2

# data artifacts and config
echo "Downloading data artifacts"
echo "[Shapefile] SGMC_preproc_default.gpkg ..."
mkdir -p $HOME/app/workdir-data/preproc/sgmc
wget https://cmaas-ta2-sri-bucket.s3.us-east-2.amazonaws.com/SGMC_preproc_default.gpkg
mv SGMC_preproc_default.gpkg $HOME/app/workdir-data/preproc/sgmc/

echo "[Deposit model] _Default_.json ..."
mkdir -p $HOME/app/workdir-data/deposit_models
wget https://cmaas-ta2-sri-bucket.s3.us-east-2.amazonaws.com/_Default_.json
mv _Default_.json $HOME/app/workdir-data/deposit_models

# envs
echo "Creating 'secrets.toml'"
cat > secrets.toml << EOF
password = "${QUERYPLOT_PWD:-CriticalMaas}"
cdr_key = "${CDR_KEY:-}"
openai_key = "${OPENAI_KEY:-}"
EOF

# docker run
echo "Docker run"
sudo docker run \
    --rm \
    -it \
    -v $HOME/app/secrets.toml:/home/ubuntu/app/sri-ta2-mappable-criteria/.streamlit/secrets.toml \
    -v $HOME/app/workdir-data:/home/ubuntu/app/workdir-data \
    -p 8501:8501 \
    mye1225/cmaas-sri-queryplot:1.2

