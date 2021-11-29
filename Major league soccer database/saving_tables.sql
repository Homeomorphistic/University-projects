-----------------------------------------------------------------------------------------------------------
---------------------------------------------SAVING DATA--------------------------------------------------

--NATIONS--
SELECT *
FROM Nations
ORDER BY nation
INTO OUTFILE 'D:/Uniwersytet Wroclawski/Semestr 9/Projekt AWF/Data/RDB/nations.csv'
CHARACTER SET cp1250
FIELDS ENCLOSED BY '' 
TERMINATED BY ';' 
LINES TERMINATED BY '\n';
--nations.csv HEADER--
nation;polish_sub

--PLAYER POSITIONS--
SELECT *
FROM Player_positions
ORDER BY type
INTO OUTFILE 'D:/Uniwersytet Wroclawski/Semestr 9/Projekt AWF/Data/RDB/player_positions.csv' 
CHARACTER SET cp1250
FIELDS ENCLOSED BY '' 
TERMINATED BY ';' 
LINES TERMINATED BY '\n';
--player_positions.csv HEADER--
position;position_eng;position_pol;type

--CLIMATE DIFFERENCES--
SELECT *
FROM Climate_differences
ORDER BY climate_1, climate_2
INTO OUTFILE 'D:/Uniwersytet Wroclawski/Semestr 9/Projekt AWF/Data/RDB/climate_differences.csv' 
CHARACTER SET cp1250
FIELDS ENCLOSED BY '' 
TERMINATED BY ';' 
LINES TERMINATED BY '\n';
--climate_differences.csv HEADER--
climate_1;climate_2;difference_1;difference_2

--TEAMS--
SELECT *
FROM Teams
INTO OUTFILE 'D:/Uniwersytet Wroclawski/Semestr 9/Projekt AWF/Data/RDB/teams.csv' 
ORDER BY team_id
CHARACTER SET cp1250
FIELDS ENCLOSED BY '' 
TERMINATED BY ';' 
LINES TERMINATED BY '\n';
--teams.csv HEADER--
team_id;name;conference;climate

--PLAYERS--
SELECT *
FROM Players
ORDER BY player_id
INTO OUTFILE 'D:/Uniwersytet Wroclawski/Semestr 9/Projekt AWF/Data/RDB/players.csv' 
CHARACTER SET cp1250
FIELDS ENCLOSED BY '' 
TERMINATED BY ';' 
LINES TERMINATED BY '\n';
--players.csv HEADER--
player_id;name;team_id;nationality;position;age;height;weight

--MATCHES--
SELECT *
FROM Matches
ORDER BY match_id
INTO OUTFILE 'D:/Uniwersytet Wroclawski/Semestr 9/Projekt AWF/Data/RDB/matches.csv' 
FIELDS ENCLOSED BY '' 
TERMINATED BY ';' 
LINES TERMINATED BY '\n';
--matches.csv HEADER--
match_id;home_id;away_id;match_date;match_time;outcome;temperature;precipitation;humidity;wind;avg_temp;avg_prec;avg_hum;avg_wind;conditions

--STATISTICS--
SELECT *
FROM Statistics
ORDER BY match_id, team_id, player_id
INTO OUTFILE 'D:/Uniwersytet Wroclawski/Semestr 9/Projekt AWF/Data/RDB/statistics.csv' 
FIELDS ENCLOSED BY '' 
TERMINATED BY ';' 
LINES TERMINATED BY '\n';
--statistics.csv HEADER--
match_id;team_id;player_id;fulltime;shots;shots_succ;passes;passes_key;passes_long;passes_long_succ;passes_short;passes_short_succ;dribbles;dribbles_succ;aerials_won;touches;interceptions;clearances


-----------------------------------------------------------------------------------------------------------
---------------------------------------------COLUMN NAMES--------------------------------------------------
select GROUP_CONCAT(CONCAT("'",COLUMN_NAME,"'"))
from INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'Players'
AND TABLE_SCHEMA = 'mls'
order BY ORDINAL_POSITION;

-----------------------------------------------------------------------------------------------------------
---------------------------------------------SPECIAL DUMPS--------------------------------------------------

--PLAYERS--
SELECT player_id, p.name, p.team_id, t.name, nationality, position, age, height, weight
FROM Players p INNER JOIN Teams t ON p.team_id = t.team_id
ORDER BY player_id
INTO OUTFILE 'D:/Uniwersytet Wroclawski/Semestr 9/Projekt AWF/Data/RDB/AWF/players.csv' 
CHARACTER SET cp1250
FIELDS ENCLOSED BY '' 
TERMINATED BY ';' 
LINES TERMINATED BY '\n';
--players.csv HEADER--
player_id;player_name;team_id;team_name;nationality;position;age;height;weight

--MATCHES--
SELECT match_id, home_id, away_id, t1.name, t2.name, match_date, match_time, outcome, 
    temperature, precipitation, humidity, wind, avg_temp, avg_prec, avg_hum, avg_wind, conditions
FROM Matches m INNER JOIN Teams t1 ON m.home_id = t1.team_id INNER JOIN Teams t2 ON m.away_id = t2.team_id
ORDER BY match_date, match_time
INTO OUTFILE 'D:/Uniwersytet Wroclawski/Semestr 9/Projekt AWF/Data/RDB/AWF/matches.csv' 
CHARACTER SET cp1250
FIELDS ENCLOSED BY '' 
TERMINATED BY ';' 
LINES TERMINATED BY '\n';
--matches.csv HEADER--
match_id;home_id;away_id;home;away;match_date;match_time;outcome;temperature;precipitation;humidity;wind;avg_temp;avg_prec;avg_hum;avg_wind;conditions

--STATISTICS--
SELECT s.match_id, match_date, match_time, t.team_id, t.name, s.player_id, p.name, fulltime, shots, shots_succ, passes, passes_key, passes_long, passes_long_succ, 
		passes_short, passes_short_succ, dribbles, dribbles_succ, aerials_won, touches, interceptions, clearances
FROM Statistics s INNER JOIN Matches m ON s.match_id = m.match_id INNER JOIN Players p ON s.player_id = p.player_id
    INNER JOIN Teams t ON s.team_id = t.team_id
ORDER BY match_id, team_id, player_id
INTO OUTFILE 'D:/Uniwersytet Wroclawski/Semestr 9/Projekt AWF/Data/RDB/AWF/statistics.csv' 
CHARACTER SET cp1250
FIELDS ENCLOSED BY '' 
TERMINATED BY ';' 
LINES TERMINATED BY '\n';
--statistics.csv HEADER--
match_id;match_date;match_time;team_id;team_name;player_id;player_name;fulltime;shots;shots_succ;passes;passes_key;passes_long;passes_long_succ;passes_short;passes_short_succ;dribbles;dribbles_succ;aerials_won;touches;interceptions;clearances
