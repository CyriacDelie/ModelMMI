using OrdinaryDiffEq, Plots, ComponentArrays, DelimitedFiles, ForwardDiff
include("import_data.jl")
include("model_func.jl")

# Choix du regime
global PART = 2
# Choix des compartiments a analyser (relatifs a la tranche d'age AGE_COMP)
global COMP  = ["Iasym","Imild","Isev"]
global AGE_COMP = 1
# Choix du parametre variant (relatifs a la tranche d'age AGE_PARAM)
global PARAM = "qasym"
global AGE_PARAM = 2
# Pas d'evaluation
global step = 0.25

## Gestion des indices / import des donnees
PARAM_idx = PARAM_DICT[PARAM]
COMP_idx  = get.([COMP_DICT],COMP,0)
BASE_PARAM = load_parameters("param_p"*string(PART))
STATE_INIT = load_state_init("state_init_"*string(PART))

CONTACT = (import_contact(3) .* (POPULATIONS ./ POPULATION_TOTAL))
CONTACT = (CONTACT + CONTACT')./2

MODEL_ODE(dSTATE,STATE,PARAM,t) = SIMID_func(dSTATE,STATE,PARAM,t,CONTACT)
PROB = ODEProblem(ODE_FUNC, STATE_INIT, T_SPANS[PART], BASE_PARAM)
if PARAM_idx < 15
    idx = 3*(PARAM_idx-1) + AGE_PARAM
else
    idx = 3*14 + (PARAM_idx - 14)
end

BASE_SOL = stack(solve(PROB,Rosenbrock23(autodiff=true), p = BASE_PARAM,saveat=step).u)[AGE_COMP,COMP_idx,:]

# fonction d'evaluation sur un seul parametre (les autres parametres sont fixes)
function fixed_param_solution(param,ALT_PARAM,PARAM_MOD,AGE_COMP,comp_idx)
    return stack(solve(PROB,Rosenbrock23(autodiff=true), p = ALT_PARAM + param .* PARAM_MOD,saveat=step).u)[AGE_COMP,comp_idx,:]
end

# Definition du jeu de parametre fixe et du parametre variant
# (permet la definition de fixed_param_solution sans allouer de memoire donc permet l'autodifferentiation)

ALT_PARAM = load_parameters("param_p"*string(PART))
ALT_PARAM[idx] = 0
PARAM_MOD = zeros(size(ALT_PARAM))
PARAM_MOD[idx] = 1

# Parametre de plot

draw_T_SPAN = T_SPANS[PART][1]:step:T_SPANS[PART][2]

plot(title = "Sensibilité des compartiments à "*string(PARAM)*" (régime "*string(PART)*")",
     titlefontsize = 20,
     legend=:outerbottom,
     legendcolumns=3,
     legendfontsize = 12,
     palette =:Accent_3,
     size = (1600,800))

global n = 0

# Evaluation de la derivee partielle (par rapport au parametre variant) de la solution a tout instant 

for comp_idx = COMP_idx
    global n = n + 1
    sol_fun(param) = fixed_param_solution(param,ALT_PARAM,PARAM_MOD,AGE_COMP,comp_idx)
    global sens_grad = ForwardDiff.jacobian(sol_fun, [BASE_PARAM[idx]])
    plot!(draw_T_SPAN,sens_grad,label = COMP[n]*" (sensibilité)",linewidth=4.5)
end

# Plot des solutions

for j = 1:length(COMP)
    plot!(draw_T_SPAN,BASE_SOL[j,:],label = COMP[n]*" (solution)",linewidth=4.5,linestyle = :dash)
end

display(plot!())
