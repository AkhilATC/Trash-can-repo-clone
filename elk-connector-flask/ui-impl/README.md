## PDF Generator

#### Setup
##### Setup loacally
```commandline
git clone https://github.com/AkhilATC/EquipoAssign.git
cd ui-impl
python3 app.py
```
##### Docker
```commandline
git clone https://github.com/AkhilATC/EquipoAssign.git
cd ui-impl
docker build --tag pdf-gen-app .
docker run -d -p 5000:5000 pdf-gen-app
```