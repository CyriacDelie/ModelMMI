Construction d'un modèle (à temps continu) de la pandémie de COVID-19 en Finlande de février à décembre 2020.

## Structure du modèle

La structure générale du modèle est inspirée de celle de SIMID. De plus, trois classes d'âge sont considérées : 
- Moins de 25 ans.
- De 25 à 64 ans.
- 65 ans et plus.
Pour chacune des classes d'âge, le modèle simule de façon continue l'évolution des compartiments :
- S : Susceptibles.
- E : Exposés.
- I_presym : Infectés pré-symptomatiques.
- I_asym : Infectés asymptomatiques.
- I_mild : Infectés avec symptômes usuels.
- I_sev : Infectés avec symptômes sévères.
- I_hosp : Infectés hospitalisés.
- I_icu : Infectés en soins intensifs.
- R : Rétablis.
- D : Décédés.

Les infections se font en considérant les quantités I_presym, I_asym, I_mild et I_sev pondérés par trois paramètres q_presym, q_asym et q_sym (pour les infections usuelles et sévères) et la matrice des contacts sociaux.
L'infectiosité des individus hospitalisés et en soins intensifs est négligée. Aussi, la susceptibilité de chacune des tranches d'âges est modulée par le vecteur de paramètres q_age. 
Le reste des transitions entre compartiment est géré de façon similaire à SIMID.

## Données

Les données suivantes sont utilisées lors de l'entraînement du modèle :
- Nouveaux cas testés hebdomadaires par tranches d'âges (Source : ECDC, https://www.ecdc.europa.eu/en/covid-19/data) ("Données 1").
- Cas actifs hebdomadaires par tranches d'âges (Source : Institut national de la santé (Finlande), https://sampo.thl.fi/pivot/prod/en/epirapo) ("Données 2").
- Patients hospitalisés (actifs), en soins intensifs (actifs) et décès (totaux) quotidiens (Source : Our World in Data, https://ourworldindata.org) ("Données 3").
L'entraînement se fera sur les données hebdomadaires, les données quotidiennes sont converties accordément.
La matrice des contacts sociaux utilisée provient de http://www.socialcontactdata.org.

## Implémentation

L'implémentation du modèle a été effectuée en Julia (v1.8.5). L'entraînement se fait via la minimisation d'une fonction perte par méthode de descente.
La mise à jour des paramètres est faite par la méthode ADAM de la librairie Flux (avec un taux d'apprentissage dépendant de la situation) et les informations relatives au gradient de la fonction de perte sont obtenues par différentiation automatique (différentiation avant de la librairie ForwardDiff / SciML). Nous utilisons l'intégrateur "Rosenbrock23" de la librairie DifferentialEquations pour toute intégration numérique.

La fonction de perte du modèle est construite comme suivant : 
- Nous souhaitons minimiser la somme des quantités (sqrt(simulation) - sqrt(données))^2 pour chacune des données mise à disposition et chaque état simulé correspondant.
- Les données 2 sont comparées par tranche d'âge avec la somme des compartiments I_presym, I_asym, I_mild, I_sev, I_hosp et I_icu.
- Les données 3 sont comparées (respectivement) avec les compartiments I_hosp, I_icu et D, sommés sur les tranches d'âge.
- Les données 1 sont comparées, pour plus de facilités, avec un compartiment factice qui cumule la totalité des infectés du modèle (à chaque passage de S à E, il gagne la même quantité sans jamais rien perdre).
- Nous pouvons pondérer les différentes composantes de cette fonction de pertes afin d'éviter, par exemple, que la perte sur un compartiment avec peu d'individus soit négligée ou que la perte liée à un compartiment soit plus fortement pénalisée.
- Nous pouvons également ajouter un coefficient de proportionnalité sur les données afin de laisser le modèle surévaluer (légèrement) les quantités d'individus dans les compartiment (utile pour le début de la pandémie).

L'entraînement s'effectue sur deux boucles imbriquées. La boucle intérieure correspond à l'entraînement usuel par la méthode de descente. La boucle extérieure opère certains changement afin de faciliter la gestion des paramètres et d'éviter les solutions dégénérées. Entre autres :
- Les paramètres qui auraient dépassé une valeur limite sont ramenés à une valeur convenable. En particulier les paramètres correspondant à des probabilités sont ramenés entre 0 et 1. Cela permet également d'éviter l'exclusion de compartiments.
- La plupart des groupes de trois paramètres correspondant aux tranches d'âge sont amenés à rester du même ordre de grandeur, afin d'éviter un comportement trop hétérogène du modèle.

Trois entraînements sont nécessaires afin de déterminer les jeux de paramètres correspondant aux trois différents régimes adoptés en réaction à la pandémie en Finlande :
- Semaines 7 à 13 : Début de la pandémie, aucune mesure.
- Semaines 13 à 24 : Confinement et mesures de sécurité.
- Semaines 24 à 50 : Arrêt du confinement et maintient des mesures de sécurité.
Les entraînements sont faits séparément, l'un après l'autre. Une partie du modèle entraînée fournit la condition initiale de la partie suivante.

Les différents parties du code sont réparties dans les fichiers :
- import_data.jl : Toutes fonctions relatives à l'importation des données dans le programme. Contient également quelques fonctions utilitaires et des définitions de variables générales.
- model_func.jl : Contient la fonction du problème de Cauchy relatif au modèle.
- model_training.jl : Script d'entraînement du modèle. La version du script fournie effectue une démonstration de l'entraînement sur le troisième régime à partir d'un jeu de paramètres initiaux relativement quelconque.
- simulation_results.jl : Script pour la simulation du modèle et les différents graphes résultant. Les résultats sont disponibles dans le dossier PICS.
- draw_sensibility.jl : Script pour visualiser (en partie) l'analyse de sensibilité.

Ce dernier script calcule, pour un certain paramètre et certain(s) compartiment(s), la sensibilité du compartiment au paramètres sous la forme de la dérivée partielle de la solution par rapport au paramètre, évaluée à différents instants (essentielle pour les méthodes locales d'analyse de sensibilité, ici obtenue par différentiation automatique ForwardDiff). Ce sont certaines de ces quantités qui sont utilisées pour obtenir le gradient de la fonction de perte et d'appliquer la méthode de descente.