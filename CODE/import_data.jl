using CSV, DataFrames, DelimitedFiles, Statistics

# fonction d'import des donnees ECDC
function import_data(country::String,age_group::String)
    DATA = DataFrame(CSV.File("./DATA/ECDC.csv"))
    DATA = DATA[!,[:country,:year_week,:age_group,:new_cases,:population]]

    find_country(c::String15) = c == country
    DATA = filter(:country => find_country,DATA)
    
    if age_group == "all_fused"
        DATA = DATA[!,[:country,:year_week,:new_cases,:population]]
        population_total = sum(abs,unique(DATA[!,:population]))
        combine(groupby(DATA, :year_week), :new_cases .=> sum, renamecols=false)
        DATA[!,:population] .= population_total
    else
        find_age_group(ag::String7) = ag == age_group
        DATA = filter(:age_group => find_age_group,DATA)
    end

    return DATA
end

# fonction d'import de la matrice de contact sociaux
function import_contact(n)
    return readdlm("./DATA/social_contact_matrix_"*string(n)*".csv", ',', Float64)
end

# fonction d'import des donnees OWID et THL
function import_hosp_data(location::String)
    DATA = DataFrame(CSV.File("./DATA/owid-covid-data_icu_hosp.csv"))
    DATA = DATA[!,[:location,:icu_patients,:hosp_patients,:people_fully_vaccinated,:total_deaths]]

    find_country(c::String) = c == location
    DATA = filter(:location => find_country,DATA)

    DATA_hosp = DATA[!,:hosp_patients]
    DATA_icu = DATA[!,:icu_patients]
    DATA_vaccinated = DATA[!,:people_fully_vaccinated]
    DATA_deceased   = DATA[!,:total_deaths]

    # Conversion en donnees hebdomadaires
    weeks = trunc(Int,length(DATA_hosp)/7)

    DATA_hosp_weekly = zeros(1,weeks)
    DATA_icu_weekly = zeros(1,weeks)
    DATA_vaccinated_weekly = zeros(1,weeks)
    DATA_deceased_weekly = zeros(1,weeks)

    for i = 1:weeks-1

        hosp = mean(skipmissing(DATA_hosp[3+7*i:3+7*(i+1)]))
        icu  = mean(skipmissing(DATA_icu[3+7*i:3+7*(i+1)]))

        vaccinated = skipmissing(DATA_vaccinated[3+7*i:3+7*(i+1)])
        deceased = skipmissing(DATA_deceased[3+7*i:3+7*(i+1)])

        if ~isempty(vaccinated)
            vaccinated = maximum(vaccinated)
        else
            vaccinated = 0
        end

        if ~isempty(deceased)
            deceased = maximum(deceased)
        else
            deceased = 0
        end

        if isnan(hosp)
            hosp = 0
        end
        if isnan(icu)
            icu = 0
        end
        if isnan(vaccinated)
            vaccinated = 0
        end
        if isnan(deceased)
            deceased = 0
        end

        DATA_hosp_weekly[i+1] = hosp
        DATA_icu_weekly[i+1] = icu
        DATA_vaccinated_weekly[i+1] = vaccinated
        DATA_deceased_weekly[i+1] = deceased
    end

    age_classes = ["00-10","10-20","20-30","30-40","40-50","50-60","60-70","70-80","80-","All ages"]
    DATA_total_cases = zeros(178,10)
    DATAFRAME_total_cases  = DataFrame(CSV.File("./DATA/fact_epirapo_covid19case.csv"))
    DATAFRAME_total_cases  = DATAFRAME_total_cases[!,[:age,:val]]

    for i = 1:10
        age_class = age_classes[i]
        find_age(age::String15) = age == age_class
        DATA_age = filter(:age => find_age,DATAFRAME_total_cases)
        DATA_total_cases[:,i] = DATA_age[!,:val]
    end

    DATA_total_cases_weekly = zeros(178,4)
    DATA_total_cases_weekly[:,1] = sum(DATA_total_cases[:,1:2];dims=2) + DATA_total_cases[:,3]./2
    DATA_total_cases_weekly[:,2] = DATA_total_cases[:,3]./2 + sum(DATA_total_cases[:,4:6];dims=2) + DATA_total_cases[:,7]./2
    DATA_total_cases_weekly[:,3] = DATA_total_cases[:,7]./2 + sum(DATA_total_cases[:,8:9];dims=2)
    DATA_total_cases_weekly[:,4] = DATA_total_cases[:,10]

        
    

    return DATA_hosp_weekly, DATA_icu_weekly, DATA_total_cases_weekly, DATA_vaccinated_weekly, DATA_deceased_weekly
end

# Importe l'ensemble des donnees et fait les modifications necessaires a l'usage lors de l'entrainement et autres
# (les noms DATA_new_cases et DATA_total_cases ne sont pas super judicieux)
function data_load(COUNTRY,POPULATION_TOTAL,START,T_END)
    AGE_GROUPS = ["<15yr","15-24yr","25-49yr","50-64yr","65-79yr","80+yr"]

    n_age = 6
    DATA_new_cases = zeros(n_age ,T_END+1)
    POPULATIONS = zeros(n_age,1)

    for i in eachindex(AGE_GROUPS)
        DATA_age_group = import_data(COUNTRY,AGE_GROUPS[i])
        POPULATIONS[i] = DATA_age_group[!,:population][1]
        DATA_age_group = DATA_age_group[!,:new_cases]
        DATA_new_cases[i,:] = DATA_age_group[START+1:START+T_END+1]
    end

    DATA_new_cases[1,:] = sum(DATA_new_cases[[1,2],:];dims=1)
    DATA_new_cases[2,:] = sum(DATA_new_cases[[3,4],:];dims=1)
    DATA_new_cases[3,:] = sum(DATA_new_cases[[5,6],:];dims=1)
    DATA_new_cases = DATA_new_cases[1:3,:]

    POPULATIONS[1] = sum(POPULATIONS[1:2])
    POPULATIONS[2] = sum(POPULATIONS[3:4])
    POPULATIONS[3] = sum(POPULATIONS[5:6])
    POPULATIONS = POPULATIONS[1:3]

    n_age = 3

    DATA_hosp, DATA_icu, DATA_total_cases, DATA_vaccinated, DATA_deceased = import_hosp_data(COUNTRY)
    DATA_hosp = DATA_hosp[START:START+T_END]
    DATA_icu  = DATA_icu[START:START+T_END]
    DATA_total_cases = DATA_total_cases[START:START+T_END,:]
    DATA_vaccinated = DATA_vaccinated[START:START+T_END]
    DATA_deceased = DATA_deceased[START:START+T_END]


    cumsum!(DATA_new_cases, DATA_new_cases; dims = 2)
    DATA_new_cases = DATA_new_cases .- DATA_new_cases[:,1]

    CONTACT = (import_contact(3) .* (POPULATIONS ./ POPULATION_TOTAL))
    CONTACT = (CONTACT + CONTACT')./2

    return DATA_new_cases, DATA_hosp, DATA_icu, DATA_total_cases, DATA_vaccinated, DATA_deceased, CONTACT, POPULATIONS
end

# chargement d'un jeu de parametre depuis un CSV et placement dans un ComponentArray
function load_parameters(name)
    PARAM_MATRIX = readdlm("./PARAMETERS/"*name*".csv", ',', Float64)
    PARAM =  ComponentArray(gamma  = PARAM_MATRIX[1,:],
                                theta  = PARAM_MATRIX[2,:],
                                delta1 = PARAM_MATRIX[3,:],
                                delta2 = PARAM_MATRIX[4,:],
                                psi    = PARAM_MATRIX[5,:],
                                omega  = PARAM_MATRIX[6,:],
                                delta3 = PARAM_MATRIX[7,:],
                                delta4 = PARAM_MATRIX[8,:],
                                tau1   = PARAM_MATRIX[9,:],
                                tau2   = PARAM_MATRIX[10,:],
                                p      = PARAM_MATRIX[11,:],
                                phi    = PARAM_MATRIX[12,:],
                                v2     = PARAM_MATRIX[13,:],
                                qage   = PARAM_MATRIX[14,:],
                                qpresym= PARAM_MATRIX[15,1],
                                qasym  = PARAM_MATRIX[15,2],
                                qsym   = PARAM_MATRIX[15,3])
    return PARAM
end
    
# chargement d'une matrice de conditions initiales depuis un CSV
function load_state_init(name)
    STATE_INIT = readdlm("./PARAMETERS/"*name*".csv", ',', Float64)
    STATE_INIT[:,11] .= 0
    return ceil.(STATE_INIT)
end

# Variables générales pour le code

COUNTRY = "Finland"
POPULATION_TOTAL = 5540745
POPULATIONS = [1467503,2810352,1255938]
n_age = 3

param_as_matrix(PARAM) = reshape(PARAM[1:15*n_age],n_age,15)'

PARAM_DICT = Dict([("gamma"  ,1),
                   ("theta"  ,2),
                   ("delta1" ,3),
                   ("delta2" ,4),
                   ("psi"    ,5),
                   ("omega"  ,6),
                   ("delta3" ,7),
                   ("delta4" ,8),
                   ("tau1"   ,9),
                   ("tau2"   ,10),
                   ("p"      ,11),
                   ("phi"    ,12),
                   ("qage"   ,14),
                   ("qpresym",15),
                   ("qasym"  ,16),
                   ("qsym"   ,17)])


COMP_DICT = Dict([("S"      ,1),
                  ("E"      ,2),
                  ("Ipresym",3),
                  ("Iasym"  ,4),
                  ("Imild"  ,5),
                  ("Isev"   ,6),
                  ("Ihosp"  ,7),
                  ("Iicu"   ,8),
                  ("R"      ,9),
                  ("D"      ,10),
                  ("Itot"   ,11)])

T_SPANS = [(7, 13),(13, 24),(24, 50)]

distance(x) = sqrt.(abs.(x))