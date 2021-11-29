---------------TABLES------------

--NATIONS--
CREATE TABLE Nations(
    nation VARCHAR(50) PRIMARY KEY,
    polish_sub VARCHAR(50) UNIQUE NOT NULL
);

--PLAYER POSITIONS--
CREATE TABLE Player_positions(
    position VARCHAR(3) PRIMARY KEY,
    position_eng VARCHAR(3) NULL UNIQUE,
    position_pol VARCHAR(3) NULL UNIQUE,
    type VARCHAR(3) NOT NULL
);

--CLIMATE--
CREATE TABLE Climate_differences(
    climate_1 VARCHAR(3) NOT NULL,
    climate_2 VARCHAR(3) NOT NULL,
    difference_1 DECIMAL(2,1) NOT NULL,
    difference_2 INT NOT NULL,
    PRIMARY KEY(climate_1, climate_2)
);

--TEAMS--
CREATE TABLE Teams(
    team_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    conference VARCHAR(20) NOT NULL CHECK (conference IN ('EASTERN', "WESTERN")),
    climate VARCHAR(3) NOT NULL
);

--PLAYERS--
CREATE TABLE Players(
    player_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    team_id INT NOT NULL,
    nationality VARCHAR(50) NOT NULL,  
    position VARCHAR(3) NOT NULL
    age INT CHECK(0<age AND age<100) NOT NULL,
    height INT CHECK(0<height AND height<250) NOT NULL,
    weight INT CHECK(0<weight AND weight<150) NOT NULL,
    FOREIGN KEY (team_id) REFERENCES Teams(team_id) ON UPDATE CASCADE ON DELETE RESTRICT,
    FOREIGN KEY (nationality) REFERENCES Nations(nation) ON UPDATE CASCADE ON DELETE RESTRICT,
    FOREIGN KEY (position) REFERENCES Player_positions(position) ON UPDATE CASCADE ON DELETE RESTRICT
);

--MATCHES--
CREATE TABLE Matches(
    match_id INT AUTO_INCREMENT PRIMARY KEY,
    home_id INT NOT NULL,
    away_id INT NOT NULL,
    match_date DATE NOT NULL,
    match_time TIME, #NOT NULL
    outcome INT CHECK (outcome IN (0,1,3)) NOT NULL,
    temperature INT CHECK(0<temperature AND temperature<150) NOT NULL, 
    precipitation DEC(3,2) CHECK(0<=precipitation AND precipitation<100), #NOT NULL
    humidity INT CHECK(0<=humidity AND humidity<=100) NOT NULL, 
    wind INT CHECK(0<=wind AND wind<=250) NOT NULL, 
    avg_temp DEC(5,2) CHECK(0<avg_temp AND avg_temp<150) NOT NULL,
    avg_prec DEC(4,2) CHECK(0<=avg_prec AND avg_prec<100) ,#NOT NULL
    avg_hum DEC(4,2) CHECK(0<=avg_hum AND avg_hum<=100) NOT NULL, 
    avg_wind DEC(5,2) CHECK(0<=avg_wind AND avg_wind<=250) NOT NULL, 
    conditions VARCHAR(50) NOT NULL, 
    UNIQUE (home_id, away_id, match_date, match_time),
    FOREIGN KEY (home_id) REFERENCES Teams(team_id) ON UPDATE CASCADE ON DELETE RESTRICT,
    FOREIGN KEY (away_id) REFERENCES Teams(team_id) ON UPDATE CASCADE ON DELETE RESTRICT
);

--STATISTICS--
CREATE TABLE Statistics(
    match_id INT NOT NULL,
    team_id INT NOT NULL, 
    player_id INT NOT NULL,
    fulltime BOOL NOT NULL,
    shots INT NOT NULL,
    CONSTRAINT CHK_statistics_shots CHECK(0<=shots AND shots<100),
    shots_succ INT  NOT NULL, 
    CONSTRAINT CHK_statistics_shots_succ CHECK(0<=shots_succ AND shots_succ<=shots),
    passes INT NOT NULL,
    CONSTRAINT CHK_statistics_passes CHECK(0<=passes AND passes<300),
    passes_key INT  NOT NULL,
    CONSTRAINT CHK_statistics_passes_key CHECK(0<=passes_key AND passes_key<=passes),
    passes_long INT NOT NULL,
    #CONSTRAINT CHK_statistics_passes_long CHECK(0<=passes_long AND passes_long<=passes),
    passes_long_succ INT NOT NULL,
    #CONSTRAINT CHK_statistics_passes_long_succ CHECK(0<=passes_long_succ AND passes_long_succ<=passes_long),
    passes_short INT NOT NULL,
    #CONSTRAINT CHK_statistics_passes_short CHECK(0<=passes_short AND passes_short<=passes),
    passes_short_succ INT NOT NULL,
    CONSTRAINT CHK_statistics_passes_short_succ CHECK(0<=passes_short_succ AND passes_short_succ<=passes_short),
    dribbles INT NOT NULL,
    CONSTRAINT CHK_statistics_dribbles CHECK(0<=dribbles AND dribbles<100),
    dribbles_succ INT NOT NULL,
    #CONSTRAINT CHK_statistics_dribbles_succ CHECK(0<=dribbles_succ AND dribbles_succ<dribbles),
    aerials_won INT NOT NULL,
    CONSTRAINT CHK_statistics_aerials CHECK(0<=aerials_won AND aerials_won<100),
    touches INT NOT NULL,
    CONSTRAINT CHK_statistics_touches CHECK(0<=touches AND touches<300),
    interceptions INT NOT NULL,
    CONSTRAINT CHK_statistics_interceptions CHECK(0<=interceptions AND interceptions<100),
    clearances INT #NOT NULL
    CONSTRAINT CHK_statistics_clearances CHECK(0<=clearances AND clearances<100),
    #CONSTRAINT CHK_statistics_passes_total CHECK(passes = passes_short  + passes_long),
    PRIMARY KEY (match_id, player_id),
    FOREIGN KEY (match_id) REFERENCES Matches(match_id) ON UPDATE CASCADE ON DELETE RESTRICT,
    FOREIGN KEY (team_id) REFERENCES Teams(team_id) ON UPDATE CASCADE ON DELETE RESTRICT,
    FOREIGN KEY (player_id) REFERENCES Players(player_id) ON UPDATE CASCADE ON DELETE RESTRICT
);