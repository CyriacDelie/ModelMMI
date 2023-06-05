using ComponentArrays

# fonction de Cauchy de l'EDO
function SIMID_func(dSTATE,STATE,PARAM,t,CONTACT)

    dSTATE[:,11]=   PARAM.qage .* (2e-7 .* CONTACT * (PARAM.qpresym .* STATE[:,3] + PARAM.qasym .* STATE[:,4] + PARAM.qsym .* (STATE[:,5] + STATE[:,6])) .* STATE[:,1])
    #dS
    dSTATE[:,1] = - dSTATE[:,11]
    #dE
    dSTATE[:,2] = - PARAM.gamma .* STATE[:,2] + dSTATE[:,11]
    #dIpresym
    dSTATE[:,3] = - PARAM.theta   .* STATE[:,3] + PARAM.gamma   .* STATE[:,2]
    #dIasym
    dSTATE[:,4] = - PARAM.delta1  .* STATE[:,4] + PARAM.theta   .* STATE[:,3] .* PARAM.p .* 1e-1
    #dImild
    dSTATE[:,5] = - PARAM.delta2  .* STATE[:,5] - 1e-1 .* PARAM.psi     .* STATE[:,5] + PARAM.theta   .* STATE[:,3] .* (1  .- (PARAM.p .* 1e-1))
    #dIsev
    dSTATE[:,6] = - PARAM.omega   .* STATE[:,6] + 1e-1 .* PARAM.psi     .* STATE[:,5]
    #dIhosp
    dSTATE[:,7] = - PARAM.delta3  .* STATE[:,7] - PARAM.tau1    .* STATE[:,7] + PARAM.omega   .* STATE[:,6] .* PARAM.phi .* 1e-1
    #dIcu
    dSTATE[:,8] = - PARAM.delta4  .* STATE[:,8] - PARAM.tau2    .* STATE[:,8] + PARAM.omega   .* STATE[:,6] .* (1 .- (PARAM.phi .* 1e-1))
    #dR
    dSTATE[:,9] =   PARAM.delta1  .* STATE[:,4] + PARAM.delta2  .* STATE[:,5] + PARAM.delta3  .* STATE[:,7] + PARAM.delta4  .* STATE[:,8]
    #dD
    dSTATE[:,10]=   PARAM.tau1    .* STATE[:,7] + PARAM.tau2    .* STATE[:,8] 

    
end