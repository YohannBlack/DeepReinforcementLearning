
# Deep Reinforcement Learning

The goal of this project is to implement mulitple Deep Reinforcement Learning algorithm from scratch without using the OpenAi Gym library for the environments. All the Reinforcment Learning algorithms will be implemented from scratch and performances will be compared.

## Installation

To install the procject use the following commands:

```bash
  git clone https://gitlab.com/g2m-ai/demo/mini-demo-veille-concurentielle.git
```

To create a virtual environment:
#### On window :
```bash
python -m venv venv_name
.\venv_name\Scripts\activate
```
#### On Ubuntu/OSX :
```bash
python -m venv venv_name
source venv_name/bin/activate
```
To install the dependencies :
```bash
cd repo
pip install -r requirements.txt
```
## Environment Variables

To run this project, you will need to add the following environment variables to your .env file
`OPENAI_API_KEY`
`DB_PASSWORD`
`DB_USER`
`DB_PORT`
`DB_NAME`
`DB_HOST`

Your file should look like this :
```bash
OPENAI_API_KEY=sk-proj-...
DB_PASSWORD=password
DB_USER=username
DB_PORT=port_nb
DB_NAME=name
DB_HOST=hostname
```

## Scripts
- `db.py` is the script containing all functions that are databse related.
- `front.py` is the main script. This is the script you should run start everything.
- `task.py` is the script containing all webscrapping related code.
- `helper.py` is the script containing all the helper functions.
- `/database` is the folder containing the SQL script to create the tables in the databse. This script should be executed once when first creating the databse.

## Demo

To launch the project for demo you should install a SQL Database of your choice (XAMPP for MySQL, PostgreSQL, etc.). Modify the environment variable in the .env file to match with your database.
When the databse is live and running run the following command :
```bash
streamlit run front.py
```
The streamlit app will be live and you will be able to interact with the website.
You might have to put the password on the master user as the project will run a sudo command.


## TODO
🙅 = not done, ✅ = done.
- [🙅] Generate a new key when quota will be added to hello@g2m-ai.com OpenAI account.
- [🙅] Deploy the website to Azure Web App service and connect to a DB.
- [✅] Create a mapping file for the different services (Chirurgie Capilaire (100, 200, 400...) -> Chirurgie capilaire)
- [✅] Scrape multiple website and compare them to each other to know their positionning compared to each other.
- [✅] AttributeError: FetchNode object has no attribute update_state. This error happens on some website. Inquireries should be made.
- [✅] Generate a pdf from a .md file with the price comparison  

## Helpful Links
- [GitHub discussion about schema](https://github.com/ScrapeGraphAI/Scrapegraph-ai/discussions/328)
- [ScrapGraphAI Schema](https://docs.pydantic.dev/latest/examples/files/)
- [Azure deployment of streamlit app](https://medium.com/@MSufiyanGhori/how-to-use-azure-to-deploy-your-web-app-container-for-free-e11986bc3374). Uses Docker : need to check how it affects the app if we use docker for the DB too.
- [How to install XAMPP on Ubuntu](https://phoenixnap.com/kb/how-to-install-xampp-on-ubuntu)
- [How to install WAMP for Windows](https://blog.templatetoaster.com/how-to-install-wamp/)
- [ScrapeGraphAI Attribute Error](https://github.com/ScrapeGraphAI/Scrapegraph-ai/issues/762)

