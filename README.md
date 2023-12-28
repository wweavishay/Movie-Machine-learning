# Small IMDB Database
@ Contributors: Shuiling Yu, Yekai Chen, Chin-Han Lin, Zheying Lu

***
This is a movie website using `Django` as backend framework and `SQLite` as database.

This project aims to create a sample movie database including top-100 popular movie from 2008 to 2018, to build up movie box office prediction model and to show visualization radar chart for different movies.

For the current version, the following features are implemented, 
- Database insertion, deletion and update
- Search engine
- Box office **prediction** based on movie features (linear regression)
- Radar Chart **visilization** based on movie, actor and director fetures

## YouTube Demo:
https://www.youtube.com/watch?v=zLvqzsCZY94

## Application Website:
(To be added)

## Index of Contents
1. [Data and Database](#data-and-database)
2. [Box Office Prediction](#prediction)
3. [Visualization](#visualization)
4. [Deployment Instructions](#deployment-instructions)


<a name="data-and-database"></a>

## Data and Database 
Movie, actor and director information from [IMDB](https://www.imdb.com) are the origin data source. Relevant data are crawled using packages such as [beautifulsoup](https://pypi.org/project/beautifulsoup4/).

Currently, there are about **1100 movies** in the database, which are the top-100 popular movies from 2008 to 2018.

To make it easy for development and deployment, SQLite is chosen as database. The database file is `db.sqlite3` in the `movieDB` directory.

The Entity Relation Diagram (ER Diagram) of the database is shown below.

![ER Diagram](Archive/ReadMe_Images/1_ERD.png)

<a name="prediction"></a>

## Box Office Prediction

- **Linear regression model**: A linear regression model was built based on different movie features, model paramter and evaluation matrics are recorded.
- **Box office prediction**: Based on the model, predicted and actual box office are compared.


<a name="visualization"></a>

## Visualization

An **item-based** radar chart is implemented.

Movie features, actor and director information are gathered based on specified movie name, then they are normalized to draw the radar chart. Through the radar chart, user can explore the effect of different aspects to a movie.

<a name="deployment-instructions"></a>

## Deployment Instructions
1. Install [**Python 3**](https://www.python.org/) in your computer, and make sure to set environment variable correctly.
2. Install **Django** for the Python environment. The easiest way is to use pip by running `pip install django`.
3. Change directory to application directory: `cd movieDB`
4. Open a terminal, input command: `python manage.py runserver`
5. Go to the local host address: http://127.0.0.1:8000/

