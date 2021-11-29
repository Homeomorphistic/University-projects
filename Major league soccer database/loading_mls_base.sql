--Checking filepaths--
SHOW VARIABLES LIKE "secure_file_priv";

--Starting database in MySQL--
DROP DATABASE mls;
CREATE DATABASE mls;
SHOW DATABASE;
USE mls;

--Creating mls base--
DROP TABLE mls_base;

CREATE TABLE mls_base(
	id INT,
	match_date DATE,
    match_time TIME, 
    conference VARCHAR(50), 
    team VARCHAR(50),
    home BOOL, 
    outcome VARCHAR(1), 
    score INT, 
    player_name VARCHAR(50), 
    player_position VARCHAR(50), 
    player_fulltime BOOL,
    player_nationality VARCHAR(50), 
    player_age INT, 
    player_weight INT, 
    player_height INT, 
    home_climate VARCHAR(50), 
    team_climate  VARCHAR(50), 
    difference_climate VARCHAR(50),
    difference_climate2 INT, 
    `temperature[F]` INT, 
    `precipitation[inch]` FLOAT(3), 
    `humidity[%]` INT, 
    `wind[mph]` INT, 
    `avg_temp_home[F]` FLOAT(7),
    `avg_precipitation_home[inch]` FLOAT(10), 
    avg_humidity_home FLOAT(7), 
    `avg_wind_home[mph]` FLOAT(10), 
    conditions VARCHAR(50), 
    shots INT, 
    shots_ot INT, 
    key_passes INT,
    passes_accuracy FLOAT(5), 
    aerials_won INT, 
    touches INT, 
    dribbles INT, 
    dribbles_successful INT, 
    interceptions INT, 
    clearances INT,
    passes INT, 
    long_passes INT, 
    long_passes_successful INT, 
    long_passes_accuracy FLOAT(5), 
    short_passes INT, 
    short_passes_successful INT,
    short_passes_accuracy DEC(5,2)
);

SHOW TABLES;
SHOW COLUMNS FROM mls_base;

--Importing .csv--
LOAD DATA INFILE "D:/Uniwersytet Wroclawski/Semestr 9/Projekt AWF/Data/mls_base.csv"
INTO TABLE mls_base 
CHARACTER SET cp1250
FIELDS TERMINATED BY ';' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;
---------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------
