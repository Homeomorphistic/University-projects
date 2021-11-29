DROP TABLE Nations;
DROP TABLE Player_positions;
DROP TABLE Climate_differences;
DROP TABLE Teams;
DROP TABLE Players;
DROP TABLE Matches;
DROP TABLE Statistics;
-----------------------------------------------------------------------------------------------------------
---------------------------------------------LOADING DATA--------------------------------------------------

--NATIONS--
LOAD DATA INFILE 'D:/Uniwersytet Wroclawski/Semestr 9/Projekt AWF/Data/RDB/nations.csv'
INTO TABLE Nations
CHARACTER SET cp1250
FIELDS TERMINATED BY ';' 
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(nation,polish_sub);

--PLAYER POSITIONS--
LOAD DATA INFILE 'D:/Uniwersytet Wroclawski/Semestr 9/Projekt AWF/Data/RDB/player_positions.csv'
INTO TABLE Player_positions
CHARACTER SET cp1250
FIELDS TERMINATED BY ';' 
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(position,position_eng,position_pol,type);

--CLIMATE DIFFERENCES--
LOAD DATA INFILE 'D:/Uniwersytet Wroclawski/Semestr 9/Projekt AWF/Data/RDB/climate_differences.csv'
INTO TABLE Climate_differences
CHARACTER SET cp1250
FIELDS TERMINATED BY ';' 
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(climate_1, climate_2, difference_1, difference_2);

--TEAMS--
LOAD DATA INFILE 'D:/Uniwersytet Wroclawski/Semestr 9/Projekt AWF/Data/RDB/teams.csv'
INTO TABLE Teams
CHARACTER SET cp1250
FIELDS TERMINATED BY ';' 
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(team_id,name,conference,climate);

--PLAYERS--
LOAD DATA INFILE 'D:/Uniwersytet Wroclawski/Semestr 9/Projekt AWF/Data/RDB/players.csv'
INTO TABLE Players
CHARACTER SET cp1250
FIELDS TERMINATED BY ';' 
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(player_id,name,team_id,nationality,position,age,height,weight);

--MATCHES--
LOAD DATA INFILE 'D:/Uniwersytet Wroclawski/Semestr 9/Projekt AWF/Data/RDB/matches.csv'
INTO TABLE Matches
CHARACTER SET cp1250
FIELDS TERMINATED BY ';' 
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(match_id,home_id,away_id,match_date,match_time,outcome,temperature,precipitation,
	humidity,wind,avg_temp,avg_prec,avg_hum,avg_wind,conditions);

--STATISTICS--
LOAD DATA INFILE 'D:/Uniwersytet Wroclawski/Semestr 9/Projekt AWF/Data/RDB/statistics.csv'
INTO TABLE Statistics
CHARACTER SET cp1250
FIELDS TERMINATED BY ';' 
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(match_id,team_id,player_id,fulltime,shots,shots_succ,passes,passes_key,passes_long,passes_long_succ,
	passes_short,passes_short_succ,dribbles,dribbles_succ,aerials_won,touches,interceptions,clearances);

---------INDICES--------------
ALTER TABLE Matches AUTO_INCREMENT=1;