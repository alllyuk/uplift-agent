# Digital Ocean

### How to deploy

Deployment occurs automatically after pushing to the main branch. The pipeline `.gitlab-ci.yml` and the script `scripts/deploy.sh` are responsible for the auto-deployment. 

The pipeline starts building a Docker image, then pushes it to the Gitlab Container Registry.

A deploy user is created on the DigitalOcean server to run the application on the server. The pipeline connects via SSH as the deploy user to DigitalOcean and runs the `deploy.sh` script, pulling and then running the Docker image. 

**App available here (requires VPN): [188.166.252.254:8501](http://188.166.252.254:8501/)**

### Expected prerequisites:
On the DigitalOcean server, the deploy user has the following files:
- `deploy.sh` file
- `.env` file (see `env.example` for an example)
- `artifacts/` folder

Currently, all prerequisites are dpne on the server.

### Additional notes:
- Gitlab Container Registry is configured to store the last 10 tags of each image (i.e. last 10 pushes to main). If there are more than 10 images, the oldest images will be deleted 30 days after they are uploaded.


#### Manual deploy
```bash
# local
scp -r requirements.txt Dockerfile .env sme_causal root@188.166.252.254:~

# local
ssh root@188.166.252.254
# remote
docker ps
docker kill <pid>
docker rm uplift-streamlit
curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh
mkdir -p ~/artifacts

# local
scp artifacts/synthetic_clients.csv root@188.166.252.254:~/artifacts/

# remote
cd ..
docker build -t uplift-streamlit .
docker run --env-file ~/.env -p 8501:8501 -v ~/artifacts:/app/artifacts -d --name uplift-streamlit uplift-streamlit:latest
```

Restart
```bash
docker restart uplift-streamlit
```
App available here: 188.166.252.254:8501

# Yandex.Cloud
```bash
# local
scp -r requirements.txt Dockerfile .env sme_causal itmo@89.169.179.234:~
ssh -l itmo 89.169.179.234
# remote
sudo docker build -t uplift-streamlit .

# local
mkdir -p ~/artifacts
scp artifacts/synthetic_clients.csv itmo@89.169.179.234:~/artifacts/
# remote
sudo docker stop uplift-streamlit && sudo docker rm uplift-streamlit
sudo docker run --env-file ~/.env -p 8501:8501 -v ~/artifacts:/app/artifacts -d --name uplift-streamlit uplift-streamlit:latest

89.169.179.234:8501
```