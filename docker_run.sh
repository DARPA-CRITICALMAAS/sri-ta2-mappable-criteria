docker run \
--rm \
-it \
-v /Users/e32648/Documents/CriticalMAAS/12-month_hack/mac_install/sri-ta2-mappable-criteria/.streamlit:/home/ubuntu/app/sri-ta2-mappable-criteria/.streamlit \
-v /Users/e32648/Documents/CriticalMAAS/12-month_hack/mac_install/workdir-data:/home/ubuntu/app/workdir-data \
-v /Users/e32648/Documents/CriticalMAAS/12-month_hack/mac_install/sri-ta2-mappable-criteria/config.toml:/home/ubuntu/app/sri-ta2-mappable-criteria/config.toml \
--entrypoint /bin/bash \
-p 8501:8501 \
mye1225/cmaas-sri-queryplot:1.2