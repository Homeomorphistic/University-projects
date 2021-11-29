-------------DATA LOADING---------

--NATIONS--
LOAD DATA INFILE "D:/Uniwersytet Wroclawski/Semestr 9/Projekt AWF/Data/nations.csv"
INTO TABLE Nations 
CHARACTER SET cp1250
FIELDS TERMINATED BY ';' 
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS
(polish_sub, nation);

--PLAYER_POSITIONS--
LOAD DATA INFILE "D:/Uniwersytet Wroclawski/Semestr 9/Projekt AWF/Data/player_positions.csv"
INTO TABLE Player_positions 
CHARACTER SET cp1250
FIELDS TERMINATED BY ';' 
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS;

--CLIMATE_DIFFERENCES--
INSERT INTO Climate_differences(climate_1, climate_2, difference_1, difference_2)
SELECT home_climate, team_climate, climate_diff_to_int(difference_climate), difference_climate2
FROM mls_base
GROUP BY home_climate, team_climate;

--TEAMS--
INSERT INTO Teams(name, conference, climate)
SELECT team, conference, home_climate
FROM mls_base
WHERE home = 1
GROUP BY team, conference, home_climate;

--PLAYERS--
CREATE VIEW position_dic AS
SELECT position_eng AS position_base, position, type
FROM Player_positions
UNION
SELECT position_pol AS position_base, position, type
FROM Player_positions
WHERE position_pol IS NOT NULL 
ORDER BY type;

CREATE VIEW players_view AS
SELECT m.player_name, m.team, n.nation AS nat, pd.position AS pos, m.player_age AS age, 
    m.player_height AS h, m.player_weight AS w, COUNT(*) AS occur, MAX(m.match_date) AS last_date
FROM mls_base m LEFT JOIN position_dic pd ON m.player_position = pd.position_base
    LEFT JOIN Nations n ON m.player_nationality = n.polish_sub
GROUP BY m.player_name, m.team, n.nation, pd.position_base, m.player_age, m.player_height, m.player_weight;

CREATE VIEW player_names AS
SELECT DISTINCT player_name
FROM mls_base;

INSERT INTO Players(name, team_id, nationality, position, age, height, weight)
SELECT player_name, t.team_id, most_freq_nat(player_name), most_freq_pos(player_name), 
    most_freq_age(player_name), most_freq_height(player_name), most_freq_weight(player_name)
FROM player_names pn INNER JOIN Teams t ON UPPER(last_team(pn.player_name) ) = t.name;

--MATCHES--
#BE AWARE OF 2019-09-12! (NY, SAN JOSE)
CREATE VIEW home_matches AS
SELECT match_date, match_time, score, `temperature[F]`, `precipitation[inch]`, `humidity[%]`,
       `wind[mph]`, `avg_temp_home[F]`, `avg_precipitation_home[inch]`, avg_humidity_home, 
       `avg_wind_home[mph]`, conditions, t.team_id,  MAX(id) AS max_id
FROM mls_base m INNER JOIN Teams t ON UPPER(m.team) = t.name
WHERE home = 1
GROUP BY match_date, match_time, team;

CREATE VIEW away_matches AS
SELECT match_date, t.team_id, MIN(id) AS min_id
FROM mls_base m INNER JOIN Teams t ON UPPER(m.team) = t.name
WHERE home = 0
GROUP BY match_date, match_time, team;

INSERT INTO Matches(home_id, away_id, match_date, match_time, outcome, temperature, precipitation, humidity,
                    wind, avg_temp, avg_prec, avg_hum, avg_wind, conditions)
SELECT h.team_id,  a.team_id, h.match_date, match_time, score, `temperature[F]`, `precipitation[inch]`, 
        `humidity[%]`,`wind[mph]`, `avg_temp_home[F]`, `avg_precipitation_home[inch]`, 
        avg_humidity_home, `avg_wind_home[mph]`, conditions
FROM home_matches h INNER JOIN away_matches a ON h.max_id + 1 = a.min_id;

--STATISTICS--
INSERT INTO Statistics(match_id, team_id, player_id, fulltime, shots, shots_succ, passes, passes_key, passes_long,
                        passes_long_succ, passes_short, passes_short_succ, dribbles, dribbles_succ,
                        aerials_won, touches, interceptions, clearances)
SELECT m.match_id, t.team_id, p.player_id, player_fulltime, shots, shots_ot, passes, key_passes, long_passes,
        long_passes_successful, short_passes, short_passes_successful, dribbles, dribbles_successful, 
        aerials_won, touches, interceptions, clearances
FROM mls_base mls INNER JOIN Teams t ON UPPER(mls.team) = t.name 
    INNER JOIN Matches m ON (mls.match_date = m.match_date )
    INNER JOIN Players p ON mls.player_name = p.name
WHERE (t.team_id = m.home_id AND home=1)
    OR (t.team_id = m.away_id AND home=0)
ORDER BY mls.match_date, mls.match_time;