using OrdinaryDiffEq, Plots, ComponentArrays, DelimitedFiles
include("import_data.jl")
include("model_func.jl")

POPULATION_TOTAL = 5540745

START = 7
T_END = 43

TSPAN = (START,START+T_END+1)

DATA_new_cases, DATA_hosp, DATA_icu, DATA_total_cases, DATA_vaccinated, DATA_deceased, CONTACT, POPULATIONS = data_load("Finland",POPULATION_TOTAL,START,T_END)
MODEL_ODE(dSTATE,STATE,PARAM,t) = SIMID_func(dSTATE,STATE,PARAM,t,CONTACT)
STATE_INIT = load_state_init("state_init_1")

## Integration de la solution sur les trois intervalles de temps

MODEL_PARAM_1 = load_parameters("param_p1")
MODEL_PARAM_2 = load_parameters("param_p2")
MODEL_PARAM_3 = load_parameters("param_p3")

MODEL_1 = ODEProblem(MODEL_ODE, STATE_INIT, T_SPANS[1], MODEL_PARAM_1)
SOLUTION = solve(MODEL_1,Rosenbrock23(autodiff=true),saveat=0.1)
SOL_U_1 = permutedims(stack(SOLUTION.u),[3,1,2])
SOL_T_1 = reshape(SOLUTION.t,:)

#writedlm( "./PARAMETERS/state_init_2.csv",  SOL_U_1[end,:,:], ',')

MODEL_2 = ODEProblem(MODEL_ODE, reshape(SOL_U_1[end,:,:],3,11), T_SPANS[2], MODEL_PARAM_2)
SOLUTION = solve(MODEL_2,Rosenbrock23(autodiff=true),saveat=0.1)
SOL_U_2 = permutedims(stack(SOLUTION.u),[3,1,2])
SOL_T_2 = reshape(SOLUTION.t,:)

#writedlm( "./PARAMETERS/state_init_3.csv",  SOL_U_2[end,:,:], ',')

MODEL_3 = ODEProblem(MODEL_ODE, reshape(SOL_U_2[end,:,:],3,11), T_SPANS[3], MODEL_PARAM_3)
SOLUTION = solve(MODEL_3,Rosenbrock23(autodiff=true),saveat=0.1)
SOL_U_3 = permutedims(stack(SOLUTION.u),[3,1,2])
SOL_T_3 = reshape(SOLUTION.t,:)

# Concatenation de la solution
SOL_T = reshape(cat(SOL_T_1,SOL_T_2[2:111,:,:],SOL_T_3[2:261,:,:];dims=(1,1,1)),:)
SOL_U = cat(SOL_U_1,SOL_U_2[2:111,:,:],SOL_U_3[2:261,:,:];dims=(1,1,1))

TOTAL_INFECTED_BY_AGE = reshape(sum(SOL_U[:,:,3:8];dims=3),(:,3))
TOTAL_INFECTED = sum(TOTAL_INFECTED_BY_AGE;dims=2)

# Cas actifs hebdomadaires

plot(title = "Cas actifs hebdomadaires entre février et décembre 2020",
     titlefontsize = 20,
     legend=:outerbottom,
     legendcolumns=3,
     legendfontsize = 12,
     palette =:Accent_3,
     size = (1600,800))
vline!([13,24];linewidth=2.5,linecolor = :black,linestyle = :dash,label= "Changements de régime")
plot!(SOL_T,TOTAL_INFECTED_BY_AGE,label = ["Model <25yr" "Model 25-64yr" "Model 64yr+"],linewidth=4.5)
display(scatter!(START:START+T_END,DATA_total_cases[1:T_END+1,1:3],label = ["Data <25yr" "Data 25-64yr" "Data 64yr+"],ms=6))
savefig("./PICS/weekly_active_cases.png")

# Patients hospitalises

plot(title = "Patients hospitalisés/en soins intensifs entre février et décembre 2020",
     titlefontsize = 20,
     legend=:outerbottom,
     legendcolumns=3,
     legendfontsize = 12,
     palette =:Accent_6,
     size = (1600,800))
vline!([13,24];linewidth=2.5,linecolor = :black,linestyle = :dash,label= "Changements de régime")
plot!(SOL_T,reshape(sum(SOL_U[:,:,7:8];dims=2),:,2),label = ["Model hospitalized" "Model ICU"],linewidth=4.5)
display(scatter!(START:START+T_END,[DATA_hosp,DATA_icu],label = ["Data hospitalized" "Data ICU"],ms=6))
savefig("./PICS/weekly_hosp_icu_patients.png")

# Patients hospitalises par classes d'ages

plot(title = "Patients hospitalisés par classes d'ages entre février et décembre 2020",
     titlefontsize = 20,
     legend=:outerbottom,
     legendcolumns=3,
     legendfontsize = 12,
     palette =:Accent_3,
     size = (1600,800))
vline!([13,24];linewidth=2.5,linecolor = :black,linestyle = :dash,label= "Changements de régime")
display(plot!(SOL_T,reshape((SOL_U[:,:,7]),:,3),label = ["Model <25yr" "Model 25-64yr" "Model 64yr+"],linewidth=4.5))
savefig("./PICS/weekly_hosp_icu_patients.png")

# Deces

plot(title = "Décès entre février et décembre 2020",
     titlefontsize = 20,
     legend=:outerbottom,
     legendcolumns=3,
     legendfontsize = 12,
     palette =:Accent_6,
     size = (1600,800))
vline!([13,24];linewidth=2.5,linecolor = :black,linestyle = :dash,label= "Changements de régime")
plot!(SOL_T,reshape(sum(SOL_U[:,:,10];dims=2),:,1),label = "Model deceased",linewidth=4.5)
display(scatter!(START:START+T_END,DATA_deceased,label = "Data deceased",ms=6))
savefig("./PICS/weekly_deceased.png")

# Cas totaux

plot(title = "Nombres totaux d'infections entre février et décembre 2020",
     titlefontsize = 20,
     legend=:outerbottom,
     legendcolumns=3,
     legendfontsize = 12,
     palette =:Accent_3,
     size = (1600,800))
vline!([13,24];linewidth=2.5,linecolor = :black,linestyle = :dash,label= "Changements de régime")
plot!(SOL_T,SOL_U[:,:,11],label = ["Model <25yr" "Model 25-64yr" "Model 64yr+"],linewidth=4.5)
display(scatter!(START:START+T_END,DATA_new_cases',label = ["Data <25yr" "Data 25-64yr" "Data 64yr+"],ms=6))
savefig("./PICS/weekly_total_cases.png")

# Evolution des differents compartiments infectés

plot(title = "Évolution des différents compartiments infectés (25-64 ans) entre février et décembre 2020",
     titlefontsize = 20,
     legend=:outerbottom,
     legendcolumns=3,
     legendfontsize = 12,
     palette =:Accent_6,
     size = (1600,800))
vline!([13,24];linewidth=2.5,linecolor = :black,linestyle = :dash,label= "Changements de régime")
display(plot!(SOL_T[1:371],SOL_U[1:371,2,[2,3,4,5,6,7]],label = ["Ipresym" "Iasym" "Imild" "Isev" "Ihosp" "Iicu"],linewidth=4.5))
savefig("./PICS/weekly_infected_class_2.png")