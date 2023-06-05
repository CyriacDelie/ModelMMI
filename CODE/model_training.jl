using OrdinaryDiffEq, Flux, Plots, ComponentArrays, Optimisers, ForwardDiff, Statistics
include("import_data.jl")
include("model_func.jl")

####### GENERAL PARAMETERS

# Choix du "regime"
PART = 3
# Nombre de generations dans la boucle interne
GENERATION = 100
# Nombre d'iterations dans la boucle externe
LOOP = 15
# Permet d'eviter la comparaison avec les q-1 premieres donnees dans la fonction de perte
q = 2
# Coefficient qui augmente artificiellement certaine donnees (permet de preferer la sur-evaluation a la sous-evaluation)
coef = 1.00
# Learning rate pour ADAM
lr = 0.005

####### DATA LOADING

# Periode d'integration/de lecture des donnees
T_SPAN = T_SPANS[PART]
START = T_SPAN[1]
T_END = T_SPAN[2] - T_SPAN[1]
# Chargement de toutes les donnees necessaires
DATA_new_cases, DATA_hosp, DATA_icu, DATA_total_cases, DATA_vaccinated, DATA_deceased, CONTACT, POPULATIONS = data_load(COUNTRY,POPULATION_TOTAL,START,T_END)

####### INITIALISATION

# chargement des parametres et etats initiaux a partir de fichier CSV
# (facilite les entrainements successifs)

#PARAM_init = load_parameters("param_p"*string(PART))
PARAM_init = load_parameters("param_init")
STATE_init = load_state_init("state_init_"*string(PART))

####### INTEGRATION PARAMETERS

# definition de l'EDO a integrer
PARAMS = PARAM_init
ODE_FUNC(dSTATE,STATE,PARAM,t) = SIMID_func(dSTATE,STATE,PARAM,t,CONTACT)
PROB = ODEProblem(ODE_FUNC, STATE_init, (0.0,T_END+1), PARAMS)

####### LOSS FUNCTION DEFINITION

# fonction de lecture de la solution (des compartiments necessaires)
function predict_fun(PARAM)
    return stack(solve(PROB,Rosenbrock23(autodiff=true), p = PARAM,saveat=1).u)[:,[7,8,10,11,3,4,5,6],:]
end

# fonction de perte
function loss_fun(PARAM)
    pred = predict_fun(PARAM)
    # terme relatif aux individus hospitalises
    s1 = sum(abs2, distance.(sum(pred[:,1,q:T_END+1];dims=1)) - distance.(DATA_hosp[q:T_END+1]'))
    # terme relatif aux individus en soins intensifs
    s2 = sum(abs2, distance.(sum(pred[:,2,q:T_END+1];dims=1)) - distance.(DATA_icu[q:T_END+1]'))
    # terme relatif au nombre de deces
    s3 = sum(abs2, distance.(sum(pred[:,3,q:T_END+1];dims=1)) - distance.(DATA_deceased[q:T_END+1]'))
    # terme relatif au nouveaux cas cumules (classes d'age 1 et 3)
    s4a = sum(abs2, distance.(pred[[1,3],4,q:T_END+1]) - distance.(coef .* DATA_new_cases[[1,3],q:T_END+1]);dims=[1,2])
    # terme relatif au nouveaux cas cumules (classe d'age 2)
    s4b = sum(abs2, distance.(pred[2,4,q:T_END+1]) - distance.(coef .* DATA_new_cases[2,q:T_END+1]);dims=1)
    # terme relatif au cas actifs (classes d'age 1 et 3)
    s5a = sum(abs2, reshape(distance.(sum(pred[[1,3],[1,2,5,6,7,8],q:T_END+1];dims=2)),2,:) - distance.(coef .* DATA_total_cases[q:T_END+1,[1,3]])';dims=[1,2])
    # terme relatif au cas actifs (classe d'age 2)
    s5b = sum(abs2, reshape(distance.(sum(pred[2,[1,2,5,6,7,8],q:T_END+1];dims=1)),1,:) - distance.(coef .* DATA_total_cases[q:T_END+1,2])';dims=2)
    return s1 + s2 + s3 + s4a[1] + s4b[1] + s5a[1] + s5b[1]
    #return 50*s1 + 1500*s2 + 40*s3 + 20*s4a[1] + s4b[1] + 20*s5a[1] + 10*s5b[1]
end

####### FONCTION D'AFFICHAGE

# fonction d'affichage des solutions par rapport aux donnees
function draw_sol(PARAM)

    sol = solve(PROB,Rosenbrock23(autodiff=true),p=PARAM,saveat=0.1)
    sol_u = stack(sol.u)[:,[7,8,10,11,3,4,5,6],:]

    plot(title="Age class cumulative")
    for i = 1:n_age
        plot!(sol.t, sol_u[i,4,:])
        scatter!(0:T_END, DATA_new_cases[i,:])
    end
    display(plot!())

    plot(sol.t, sum(sol_u[:,1,:];dims=1)')
    display(scatter!(0:T_END, DATA_hosp,title="Hopital (total)"))

    plot(sol.t, sum(sol_u[:,2,:];dims=1)')
    display(scatter!(0:T_END, DATA_icu,title="ICU (total)"))

    plot(sol.t, sum(sol_u[:,3,:];dims=1)')
    display(scatter!(0:T_END, DATA_deceased,title="Décès (total)"))

    plot(sol.t, reshape(sum(sol_u[:,[1,2,5,6,7,8],:];dims=2),3,:)')
    display(scatter!(0:T_END, DATA_total_cases[:,1:3],title="Cas totaux (par age)"))

    plot(sol.t, reshape(sum(sol_u[:,[1,2,5,6,7,8],:];dims=[1,2]),:,1))
    display(scatter!(0:T_END, DATA_total_cases[:,4],title="Cas totaux"))
end

####### TRAINING

global best_loss = loss_fun(PARAM_init)
global best_param = PARAM_init
global previous_param = PARAM_init

draw_sol(PARAM_init)

# boucle externe d'entrainement
for j = 1:LOOP
    local opt_state = Flux.setup(Flux.Optimisers.Adam(lr), PARAMS)
    local grad = zeros(size(PARAMS))
    print("Best loss : ")
    display(log(best_loss))
    # boucle interne
    for i = 1:GENERATION
        if mod(i-1,10) == 0
            display((j-1)*GENERATION + i-1)
        end
        # calcul du gradient et mise a jour ADAM
        grad .= ForwardDiff.gradient(loss_fun, PARAMS)
        Flux.update!(opt_state, PARAMS, grad)
    end
    # verification/correction des parametres de probabilites
    for k = 1:n_age
        if PARAMS.p[k] > 4.0 || PARAMS.p[k] < 0.01
            PARAMS.p[k] = 1.0
        end
        if PARAMS.phi[k] > 10.0 || PARAMS.phi[k] < 5.0
            PARAMS.phi[k] = 8.0
        end
    end
    # encadrement de la valeur de certains parametres
    for k = [0,1,2,3,5]
        for m = 1:3
            if PARAMS[3*k+m] > 10.0 || PARAMS[3*k+m] < 0.20
                PARAMS[3*k+m] = 4.0
            end
        end
    end
    # verification/correction des parametres q_age
    for m = 1:3
        if PARAMS[39+m] > 3.5 || PARAMS[39+m] < 0.6
            PARAMS[39+m] = 1.0
        end
    end
    # verification/correction de l'ordre de grandeur des parametres selon les tranches d'ages
    for k = [0,1,2,3,4,5,6,7,13]
        p = PARAMS[1+3*k:3*(k+1)]
        if abs(maximum(p) - minimum(p))/mean(p) > 1.50
            for m = 1:3
                PARAMS[3*k+m] = mean(p)
            end
        end
    end
    # verification/correction afin d'eviter les parametres negatifs
    for k = 1:length(previous_param)
        if PARAMS[k] < 0
            PARAMS[k] = abs(best_param[k])
        end
    end
    # calcul de la nouvelle perte
    local current_loss = loss_fun(PARAMS)
    print("Loss : ")
    display(log(current_loss))
    # mise a jour de la meilleur perte et sauvegarde des meilleurs parametres si besoin
    if current_loss < best_loss
        global best_loss = current_loss
        global best_param = PARAMS
        global best_param_matrix = param_as_matrix(best_param)
        #writedlm( "./PARAMETERS/param_p"*string(PART)*".csv",  best_param_matrix, ',')
        writedlm( "./PARAMETERS/param_log.csv",  best_param_matrix, ',')
    end
    # affichage de la nouvelle solution et des nouveaux parametres
    display(param_as_matrix(PARAMS))
    global previous_param = PARAMS
    draw_sol(PARAMS)
end