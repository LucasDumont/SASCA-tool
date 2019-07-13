#include "construction.h"

#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/inference/icm.hxx>
#include <utility>

constexpr int BASE_10 = 10;

Model modelCreation(std::map<std::string, std::size_t> const&                            fonctions,
                    std::map<std::string, std::vector<std::vector<std::vector<float>>>>& probaTab,
                    std::map<std::string, std::vector<std::size_t>> const&               link,
                    std::vector<std::size_t> const&                                      var)
{
  opengm::DiscreteSpace<> space;
  for (auto i : var) {
    space.addVariable(i);
    // std::cout <<" var "<< var[1]<<" ou var"<< var[0]<<std::endl;
  }
  Model gm(space);
  for (auto& fonction : fonctions) {
    if (fonction.second == 1) {
      bool               recherche = false;
      std::size_t        variable  = 0;
      std::vector<float> prob;
      for (auto it = link.begin(); it != link.end() || !recherche; it++) {
        if ((*it).first == fonction.first) {
          recherche = true;
          variable  = (*it).second[0];
        }
      }
      recherche = false;
      for (auto it = probaTab.begin(); it != probaTab.end() || !recherche;) {
        if ((*it).first == fonction.first) {
          recherche = true;
          prob      = (*it).second[0][0];
          it        = probaTab.erase(it);
        } else {
          ++it;
        }
      }
      fonctionPremier(gm, prob, variable);
    } else if (fonction.second == 2) {
      bool                            recherche = false;
      std::array<std::size_t, 2>      variable{0, 0};
      std::size_t                     cpt = 0;
      std::vector<std::vector<float>> prob;

      for (auto it = link.begin(); it != link.end() || !recherche; it++) {
        if (it->first == fonction.first) {
          variable[0] = it->second[0];
          variable[1] = it->second[1];
          cpt++;
          recherche = true;
        }
      }
      recherche = false;
      for (auto it = probaTab.begin(); it != probaTab.end() || !recherche;) {
        if (it->first == fonction.first) {
          recherche = true;
          prob      = (*it).second[0];
          it        = probaTab.erase(it);
        } else {
          ++it;
        }
      }
      std::array<std::size_t, 2> variableF{variable[0], variable[1]};
      fonctionSecond(gm, prob, variableF);
    } else if (fonction.second == 3) {
      bool                                         recherche = false;
      std::array<std::size_t, 3>                   variable{0, 0, 0};
      std::size_t                                  cpt = 0;
      std::vector<std::vector<std::vector<float>>> prob;
      for (auto it = link.begin(); it != link.end() || !recherche; it++) {
        if (it->first == fonction.first) {
          variable[0] = it->second[0];
          variable[1] = it->second[1];
          variable[2] = it->second[2];
          cpt++;
          recherche = true;
        }
      }
      recherche = false;
      for (auto it = probaTab.begin(); it != probaTab.end() || !recherche;) {
        if ((*it).first == fonction.first) {
          recherche = true;
          prob      = (*it).second;
          it        = probaTab.erase(it);
        } else {
          ++it;
        }
      }
      fonctionTroisieme(gm, prob, variable);
    } else {
      std::cout << "n'a pas de variable ou en a plus que 3" << std::endl;
    }
    /*
    std::cout << "nombre de fonction de degree 1 "<< gm.NrOfFunctionTypes <<
    std::endl; std::cout <<" nombre de lien  "<< gm.numberOfFactors() << std::endl;
    std::cout <<" nombre de variable  "<< gm.numberOfVariables() << std::endl;
    */
  }

  return gm;
}

void fonctionPremier(Model& gm, std::vector<float> const& prob, std::size_t var)
{
  using ExplicitFunction   = opengm::ExplicitFunction<float>;
  using FunctionIdentifier = Model::FunctionIdentifier;
  const std::array<size_t, 1> shape{gm.numberOfLabels(var)};
  ExplicitFunction            f(shape.data(), shape.data() + 1);
  for (size_t state = 0; state < prob.size(); ++state) {
    f(state) = prob[state];
    // std::cout <<" state : "<< state << " " << prob[state]<< std::endl;
  }
  // add function
  FunctionIdentifier id = gm.addFunction(f);
  // add factor
  // std::cout<<"creation de la fonction "<< var << std::endl;
  std::array<size_t, 1> variableIndex{var};
  gm.addFactor(id, variableIndex.data(), variableIndex.data() + 1);
}

void fonctionSecond(Model& gm, std::vector<std::vector<float>> const& prob, std::array<std::size_t, 2> const& var)
{ // a verif
  using ExplicitFunction   = opengm::ExplicitFunction<float>;
  using FunctionIdentifier = Model::FunctionIdentifier;
  const std::array<size_t, 2> shape{gm.numberOfLabels(var[0]), gm.numberOfLabels(var[1])};

  std::array<size_t, 2> variableIndex{};
  std::array<size_t, 2> position{};
  // add factorp
  if (var[0] < var[1]) {
    variableIndex[0] = var[0];
    variableIndex[1] = var[1];
    position[0]      = 0;
    position[1]      = 1;
  } else {
    variableIndex[0] = var[1];
    variableIndex[1] = var[0];
    position[1]      = 0;
    position[0]      = 1;
  }
  ExplicitFunction       f(shape.data(), shape.data() + 2);
  std::size_t            state;
  std::size_t            stateD;
  std::array<size_t*, 2> probaPos{&state, &stateD};
  for (state = 0; state < prob.size(); state++) {
    for (stateD = 0; stateD < prob[state].size(); stateD++) {
      if (prob.at(*probaPos.at(position.at(0))).at(*probaPos.at(position.at(1))) != 0.F) {
        f(state, stateD) = prob.at(*probaPos.at(position.at(0))).at(*probaPos.at(position.at(1)));
      }
    }
  }
  // add function
  FunctionIdentifier id = gm.addFunction(f);
  gm.addFactor(id, variableIndex.data(), variableIndex.data() + 2);
}

void fonctionTroisieme(Model&                                              gm,
                       std::vector<std::vector<std::vector<float>>> const& prob,
                       std::array<std::size_t, 3> const&                   var)
{
  using ExplicitFunction   = opengm::ExplicitFunction<float>;
  using FunctionIdentifier = Model::FunctionIdentifier;
  const std::array<size_t, 3> shape{gm.numberOfLabels(var[0]), gm.numberOfLabels(var[1]), gm.numberOfLabels(var[2])};
  // construct 3rd order function

  // sequences of variable indices need to be (and in this case are) sorted

  std::array<std::size_t, 3> variableIndexSequence{var[0], var[1], var[2]};
  std::array<std::size_t, 3> position{0, 1, 2};

  if (!(var[0] < var[1] && var[1] < var[2])) {
    for (std::size_t tmp = 0; tmp < 2; tmp++) {
      for (std::size_t a = 0; a < 2; a++) {
        if (variableIndexSequence.at(a) > variableIndexSequence.at(a + 1)) {
          std::size_t b                   = variableIndexSequence.at(a);
          variableIndexSequence.at(a)     = variableIndexSequence.at(a + 1);
          variableIndexSequence.at(a + 1) = b;
          std::size_t tmpPosition         = position.at(a);
          position.at(a)                  = position.at(a + 1);
          position.at(a + 1)              = tmpPosition;
        }
      }
    }
  }
  std::size_t            indice1 = gm.numberOfLabels(variableIndexSequence[0]);
  std::size_t            indice2 = gm.numberOfLabels(variableIndexSequence[0]);
  std::size_t            indice3 = gm.numberOfLabels(variableIndexSequence[0]);
  ExplicitFunction       f(shape.data(), shape.data() + 3, 0);
  std::size_t            state1;
  std::size_t            state2;
  std::size_t            state3;
  std::array<size_t*, 3> proba1{&state1, &state2, &state3};

  for (state1 = 0; state1 < indice1; ++state1) {
    for (state2 = 0; state2 < indice2; ++state2) {
      for (state3 = 0; state3 < indice3; ++state3) {
        if (prob.at(*proba1.at(position.at(0))).at(*proba1.at(position.at(1))).at(*proba1.at(position.at(2))) > 0.F) {
          f(state1, state2, state3) =
              prob.at(*proba1.at(position.at(0))).at(*proba1.at(position.at(1))).at(*proba1.at(position.at(2)));
        }
      }
    }
  }

  FunctionIdentifier id = gm.addFunction(f);
  gm.addFactor(id, variableIndexSequence.data(), variableIndexSequence.data() + 3);
}

void beliefPropagation(Model const& gm, std::vector<std::string>& output, std::size_t iteration, bool allVariables)
{
  using UpdateRules       = opengm::BeliefPropagationUpdateRules<Model, opengm::Maximizer>;
  using BeliefPropagation = opengm::MessagePassing<Model, opengm::Maximizer, UpdateRules, opengm::MaxDistance>;

  std::size_t const            maxNumberOfIterations = iteration;
  float const                  convergenceBound      = 1e-7F;
  double const                 damping               = 0.0;
  BeliefPropagation::Parameter parameter(maxNumberOfIterations, convergenceBound, damping);
  BeliefPropagation            bp(gm, parameter);

  // optimize (approximately)
  BeliefPropagation::VerboseVisitorType visitor;
  std::cout << "debut optimisation" << std::endl;

  bp.infer();
  std::cout << "fin optimisation" << std::endl;

  // obtain the (approximate) argmax
  std::vector<size_t> labeling(gm.numberOfVariables());
  bp.arg(labeling);
  Model::IndependentFactorType IF;
  for (size_t variable = 0; variable < gm.numberOfVariables(); ++variable) {
    bp.marginal(variable, IF);
    std::cout << "x" << variable << "=" << labeling[variable] << "\n";
    std::ostringstream trans;
    trans << variable;
    std::ostringstream trans2;
    trans2 << labeling[variable];
    std::string tempo  = "x" + trans.str();
    std::string tempo2 = "=" + trans2.str();

    output.push_back(tempo + tempo2);

    for (std::size_t j = 0; j < gm.numberOfLabels(variable); j++) {
      if (IF(j) == 0.F || allVariables) {
        std::cout << "state: " << j << " probability value: " << IF(j) << " ; ";
        std::ostringstream buff1;
        buff1 << j;
        std::string        blob = "state " + buff1.str() + "probability value :";
        std::ostringstream buff;
        buff << IF(j);
        std::string blob2 = buff.str();

        output.push_back(blob + blob2);
        std::cout << "\n";
      }
    }
  }
}

std::map<std::string, std::size_t>
transformationASM(std::vector<std::string>                                             contenue,
                  std::map<std::string, std::vector<std::vector<std::vector<float>>>>& probaTab,
                  std::map<std::string, std::vector<std::size_t>>&                     link,
                  std::vector<std::size_t>&                                            var,
                  std::map<std::string, std::size_t>&                                  fonctions,
                  std::string                                                          hammingweight,
                  std::map<std::string, std::vector<std::size_t>> const&               valeurFixer,
                  std::map<std::size_t, std::size_t>                                   valeurResultat,
                  const std::map<std::size_t, std::vector<std::size_t>>&               box,
                  bool                                                                 graph,
                  bool                                                                 cycle)
{
  std::map<std::string, std::size_t>              instruction;
  std::map<std::string, std::size_t>              traitement;
  std::map<std::string, std::vector<std::size_t>> RegY;
  std::map<std::string, std::vector<std::size_t>> RegX;
  std::map<std::string, std::vector<std::size_t>> RegZ;
  std::size_t                                     cursorRegX   = 0;
  std::size_t                                     cursorRegY   = 0;
  std::size_t                                     cursorRegZ   = 0;
  std::size_t                                     cursorLDRegX = 0;
  std::size_t                                     cursorLDRegY = 0;
  std::size_t                                     cursorLDRegZ = 0;
  std::vector<std::size_t>                        stockR30;
  std::vector<std::size_t>                        stockR28;
  std::vector<std::size_t>                        stockR26;
  std::vector<std::size_t>                        stockLPMR30;
  std::vector<std::size_t>                        stockLPMR28;
  std::vector<std::size_t>                        stockLPMR26;
  std::string                                     nameRegX;
  std::string                                     nameRegY;
  std::string                                     nameRegZ;

  instructionTaken(instruction, traitement);

  std::size_t nbLabel;
  if (graph) {
    nbLabel = 1;
  } else {
    nbLabel = 256;
  }

  std::map<std::string, std::size_t>              varIndex;
  std::map<std::size_t, std::vector<std::string>> varFonc;

  for (std::size_t i = 0; i < contenue.size(); i++) {
    bool        LPMReg   = false;
    std::size_t idLPMReg = 0;
    std::string key      = contenue[i].substr(0, contenue[i].find(' '));

    if (instruction.find(key) != instruction.end()) {
      std::string nomFonction = key;
      auto        tmpConca    = boost::lexical_cast<std::string>(i);

      nomFonction += tmpConca;
      std::vector<std::size_t> lienF;
      std::string              ligneEnCours;
      ligneEnCours          = contenue[i].substr(contenue[i].find(' ') + 1);
      std::size_t ValeurKey = instruction[key];

      if (instruction[key] > 3) {
        hammingInstructionIteration(key, ValeurKey, hammingweight, contenue, i, instruction);
      }
      if (key == "MOV") {
        hammingweight = hammingweight.substr(hammingweight.find(',') + 1);
        ValeurKey     = 1;
      }
      fonctions[nomFonction] = ValeurKey;
      std::vector<std::vector<float>> tabProbFonc;
      /*bool valueZ= false;
      bool valueX= false;
      bool valueY= false;*/

      for (std::size_t j = 0; j < ValeurKey; j++) {
        std::string nomVar;
        std::string nomVarMov;

        std::vector<float> variableProb(nbLabel);
        if (key == "MOV") {
          nomVarMov        = ligneEnCours.substr(0, ligneEnCours.find(','));
          std::size_t posi = ligneEnCours.find(',') + 2;
          nomVar           = ligneEnCours.substr(ligneEnCours.find(',') + 2, ligneEnCours.substr(0, posi).find(' '));
          if (nomVar[nomVar.size() - 1] == ' ') {
            nomVar = nomVar.substr(0, nomVar.size() - 1);
          }
          if (nomVar[nomVar.size() - 2] == ' ') {
            nomVar = nomVar.substr(0, nomVar.size() - 2);
          }
        } else if (j == 0) {
          nomVar = ligneEnCours.substr(0, ligneEnCours.find(','));
        } else if (j == 1) {
          std::size_t posi = ligneEnCours.find(',') + 2;
          nomVar           = ligneEnCours.substr(ligneEnCours.find(',') + 2, ligneEnCours.substr(0, posi).find(' '));
          if (nomVar[nomVar.size() - 1] == ' ') {
            nomVar = nomVar.substr(0, nomVar.size() - 1);
          }
          if (nomVar[nomVar.size() - 2] == ' ') {
            nomVar = nomVar.substr(0, nomVar.size() - 2);
          }
        } else {
          // a definir si utile
          /*    if(valueX){
              nomVar="R26";

          }else if(valueY){
              nomVar="R28";

          }else if(valueZ){
              nomVar="R30";

          }else{*/
          nomVar = ligneEnCours.substr(0, ligneEnCours.find(','));
          //}
        }
        // a définir si utile
        if (std::string::npos != nomVar.find('Z')) {
          //  valueZ=true;
          if (ValeurKey == 1) {
            nomVar = ligneEnCours.substr(ligneEnCours.find(',') + 2, ligneEnCours.find(' '));
            if (nomVar[nomVar.size() - 1] == ' ') {
              nomVar = nomVar.substr(0, nomVar.size() - 1);
            }
          } else {
            nomVar = "R31";
          }
        } else if (std::string::npos != nomVar.find('Y')) {
          //  valueY=true;
          if (ValeurKey == 1) {
            nomVar = ligneEnCours.substr(ligneEnCours.find(',') + 2, ligneEnCours.find(' '));
            if (nomVar[nomVar.size() - 1] == ' ') {
              nomVar = nomVar.substr(0, nomVar.size() - 1);
            }
          } else {
            nomVar = "R29";
          }
        } else if (std::string::npos != nomVar.find('X')) {
          //    valueX=true;
          if (ValeurKey == 1) {
            nomVar = ligneEnCours.substr(ligneEnCours.find(',') + 2, ligneEnCours.find(' '));
            if (nomVar[nomVar.size() - 1] == ' ') {
              nomVar = nomVar.substr(0, nomVar.size() - 1);
            }
          } else {
            nomVar = "R27";
          }
        }

        if (varIndex.find(nomVar) != varIndex.end() && (j < ValeurKey - 1 || key == "OUT" || key == "ST" ||
                                                        key == "STD" || key == "CPI" || key == "MOV" || key == "CPC")) {
          std::size_t                        id = varIndex[nomVar];
          std::map<std::string, std::size_t> vue;
          bool                               me = false;

          if (!cycle) {
            me = findCycle(varFonc, link, nomFonction, id, vue);
          } else {
            me = findLittleCycle(varFonc, nomFonction, id);
          }
          if (me) {
            std::cout << "var " << nomVar << " remplacer du a un cycle en " << nomFonction << std::endl;
            var.push_back(nbLabel);
            varIndex[nomVar] = var.size() - 1;
            lienF.push_back(var.size() - 1);
            std::vector<std::string> vec;
            vec.push_back(nomFonction);
            varFonc[var.size() - 1] = vec;
          } else {
            lienF.push_back(id);
            varFonc[id].push_back(nomFonction);
            if (key == "MOV") {
              varIndex[nomVarMov] = id;
            }
            if (key == "ST" || key == "STD") {
              std::string verif = ligneEnCours.substr(0, ligneEnCours.find(','));
              if (verif.find('Y') != std::string::npos) {
                if (!nameRegY.empty()) {
                  if (key == "STD") {
                    std::size_t index = ligneEnCours.find('+');
                    RegY[nameRegY][cursorLDRegY +
                                   std::strtoul(ligneEnCours.substr(index + 1, index + 2).c_str(), nullptr, BASE_10)] =
                        id;
                  } else {
                    RegY[nameRegY].push_back(id);
                    cursorRegY++;
                  }
                }
              } else if (verif.find('X') != std::string::npos) {
                if (!nameRegX.empty()) {
                  if (key == "STD") {
                    std::size_t index = ligneEnCours.find('+');
                    RegX[nameRegX][cursorLDRegX +
                                   std::strtoul(ligneEnCours.substr(index + 1, index + 2).c_str(), nullptr, BASE_10)] =
                        id;
                  } else {
                    RegX[nameRegX].push_back(id);
                    cursorRegX++;
                  }
                }
              } else if (verif.find('Z') != std::string::npos) {
                if (!nameRegZ.empty()) {
                  if (key == "STD") {
                    std::size_t index = ligneEnCours.find('+');
                    RegZ[nameRegZ][cursorLDRegZ +
                                   std::strtoul(ligneEnCours.substr(index + 1, index + 2).c_str(), nullptr, BASE_10)] =
                        id;
                  } else {
                    RegZ[nameRegZ].push_back(id);
                    cursorRegZ++;
                  }
                }
              }
            }
          }
        } else {
          bool        me = true;
          std::size_t id = 0;
          if (key == "LD" || key == "LDD") {
            std::string verif = ligneEnCours.substr(ligneEnCours.find(',') + 2, ligneEnCours.find(' '));
            if (verif.find('Y') != std::string::npos) {
              if (!nameRegY.empty()) {
                if (key == "LDD") {
                  std::size_t index = ligneEnCours.find('+');
                  std::size_t tmpid =
                      RegY[nameRegY][cursorLDRegY +
                                     std::strtoul(ligneEnCours.substr(index + 1, index + 2).c_str(), nullptr, BASE_10)];
                  id = tmpid;
                } else {
                  id = RegY[nameRegY][cursorLDRegY];
                  if (verif.find('+') != std::string::npos) {
                    cursorLDRegY++;
                  } else if (verif.find('-') != std::string::npos) {
                    cursorLDRegY--;
                  }
                }
              }
            } else if (verif.find('X') != std::string::npos) {
              if (!nameRegX.empty()) {
                if (key == "LDD") {
                  std::size_t index = ligneEnCours.find('+');
                  std::size_t tmpid =
                      RegX[nameRegX][cursorLDRegX +
                                     std::strtoul(ligneEnCours.substr(index + 1, index + 2).c_str(), nullptr, BASE_10)];
                  id = tmpid;
                } else {
                  id = RegY[nameRegX][cursorLDRegX];
                  if (verif.find('+') != std::string::npos) {
                    cursorLDRegX++;
                  } else if (verif.find('-') != std::string::npos) {
                    cursorLDRegX--;
                  }
                }
              }
            } else if (verif.find('Z') != std::string::npos) {
              if (!nameRegZ.empty()) {
                if (key == "LDD") {
                  std::size_t index = ligneEnCours.find('+');
                  id =
                      RegZ[nameRegZ][cursorLDRegZ +
                                     std::strtoul(ligneEnCours.substr(index + 1, index + 2).c_str(), nullptr, BASE_10)];
                } else {
                  id = RegZ[nameRegZ][cursorLDRegZ];
                  if (verif.find('+') != std::string::npos) {
                    cursorLDRegZ++;
                  } else if (verif.find('-') != std::string::npos) {
                    cursorLDRegZ--;
                  }
                }
              }
            } else {
              std::cout << "erreur sur les registre Y X Z " << std::endl;
            }
            std::map<std::string, std::size_t> vue;
            if (!cycle) {
              me = findCycle(varFonc, link, nomFonction, id, vue);
            } else {
              me = findLittleCycle(varFonc, nomFonction, id);
            }
          }
          if (!me) {
            lienF.push_back(id);
            varFonc[id].push_back(nomFonction);
            varIndex[nomVar] = id;
          } else {
            if (key == "LPM" && (nomVar == "R28" || nomVar == "R30" || nomVar == "R26")) {
              LPMReg   = true;
              idLPMReg = varIndex[nomVar];
            }
            var.push_back(nbLabel);
            varIndex[nomVar] = var.size() - 1;
            lienF.push_back(var.size() - 1);
            std::vector<std::string> vec;
            vec.push_back(nomFonction);
            varFonc[var.size() - 1] = vec;
            if (key == "SBIW") {
              std::size_t diff = std::strtoul(
                  ligneEnCours.substr(ligneEnCours.find(',') + 2, ligneEnCours.find(' ')).c_str(), nullptr, BASE_10);

              if (nomVar == "R28") {
                cursorRegY -= diff;
              } else if (nomVar == "R30") {
                cursorRegZ = -diff;
              }
            } else if (key == "ADIW") {
              std::size_t add = std::strtoul(
                  ligneEnCours.substr(ligneEnCours.find(',') + 2, ligneEnCours.find(' ')).c_str(), nullptr, BASE_10);

              if (nomVar == "R28") {
                cursorRegY += add;

              } else if (nomVar == "R30") {
                cursorRegZ += add;
              }
            }
            if (key == "LDI") {
              if (nomVar == "R26") {
                if (RegX.find(ligneEnCours.substr(ligneEnCours.find(',') + 2, ligneEnCours.find(' '))) == RegX.end()) {

                  RegX[nameRegX] = std::vector<std::size_t>(0);
                }
                nameRegX   = ligneEnCours.substr(ligneEnCours.find(',') + 2, ligneEnCours.find(' '));
                cursorRegX = 0;
              } else if (nomVar == "R28") {
                if (RegY.find(ligneEnCours.substr(ligneEnCours.find(',') + 2, ligneEnCours.find(' '))) == RegY.end()) {
                  nameRegY       = ligneEnCours.substr(ligneEnCours.find(',') + 2, ligneEnCours.find(' '));
                  cursorRegY     = 0;
                  RegY[nameRegY] = std::vector<std::size_t>(0);
                }
              } else if (nomVar == "R30") {
                if (RegZ.find(ligneEnCours.substr(ligneEnCours.find(',') + 2, ligneEnCours.find(' '))) == RegZ.end()) {
                  RegZ[nameRegZ] = std::vector<std::size_t>(0);
                }
                nameRegZ   = ligneEnCours.substr(ligneEnCours.find(',') + 2, ligneEnCours.find(' '));
                cursorRegZ = 0;
              }
            }
          }
        }

        std::string poid                    = hammingweight.substr(0, hammingweight.find(','));
        hammingweight                       = hammingweight.substr(hammingweight.find(',') + 1);
        std::size_t              value      = std::strtoul(poid.c_str(), nullptr, BASE_10);
        bool                     findValuer = false;
        std::vector<std::size_t> valueH;

        if (valeurFixer.find(nomVar) != valeurFixer.end() && (j < ValeurKey - 1 || key == "LPM")) {
          for (std::size_t f = 0; f < valeurFixer.at(nomVar).size() - 1 && !findValuer; f = f + 2) {
            if (valeurFixer.at(nomVar)[f] == i) {
              valueH.push_back(valeurFixer.at(nomVar)[f + 1]);
              findValuer = true;
            }
          }
        } else if (j == ValeurKey - 1 && valeurResultat.find(i) != valeurResultat.end()) {
          valueH.push_back(valeurResultat[i]);
          findValuer = true;
        }
        if (!findValuer) {
          if (value == 0) {
            valueH.push_back(0);
          } else {
            valueH = hammingToDecimal(value);
          }
        }
        if (nomVar == "R30" || nomVar == "R28" || nomVar == "R26" ||
            (key == "MOV" && (nomVarMov == "R30" || nomVarMov == "R26" || nomVarMov == "R28"))) {
          if (nomVar == "R30" || nomVarMov == "R30") {
            if (LPMReg) {
              stockLPMR30 = stockR30;
            }
            stockR30 = valueH;
          } else if (nomVar == "R28" || nomVarMov == "R28") {
            if (LPMReg) {
              stockLPMR28 = stockR28;
            }
            stockR28 = valueH;
          } else {
            if (LPMReg) {
              stockLPMR26 = stockR26;
            }
            stockR26 = valueH;
          }
        }
        for (std::size_t cpt = 0; cpt < nbLabel; cpt++) {
          bool trouver = false;
          for (std::size_t taille = 0; taille < valueH.size() && !trouver; taille++) {
            if (cpt == valueH[taille]) {
              variableProb[cpt] = 1.F / valueH.size();
              trouver           = true;
            }
          }
          if (!trouver) {
            variableProb[cpt] = 0;
          }
        }
        tabProbFonc.push_back(variableProb);
      }
      if (ValeurKey != 0) {
        if (ValeurKey == 3) {
          std::vector<std::vector<std::vector<float>>> tmpProba(nbLabel);
          specialInstruction(key, traitement, tmpProba, tabProbFonc, nbLabel);
          probaTab[nomFonction] = tmpProba;
        } else if (ValeurKey == 2) {
          std::vector<std::vector<std::vector<float>>> tmpProba(1);
          std::vector<std::vector<float>>              tmp(nbLabel);
          tmpProba[0] = tmp;
          for (std::size_t cpt2 = 0; cpt2 < nbLabel; cpt2++) {
            std::vector<float> tmp2(nbLabel);
            tmpProba[0][cpt2] = tmp2;
            for (std::size_t cpt3 = 0; cpt3 < nbLabel; cpt3++) {
              tmpProba[0][cpt2][cpt3] = tabProbFonc[0][cpt2] * tabProbFonc[1][cpt3];
            }
          }
          probaTab[nomFonction] = tmpProba;
        } else {
          std::vector<std::vector<std::vector<float>>> tmpProba(1);
          standartInstruction(key,
                              tmpProba,
                              nbLabel,
                              fonctions,
                              nomFonction,
                              tabProbFonc,
                              stockR30,
                              stockR28,
                              stockR26,
                              stockLPMR30,
                              stockLPMR28,
                              stockLPMR26,
                              varIndex,
                              varFonc,
                              i,
                              LPMReg,
                              idLPMReg,
                              ligneEnCours,
                              box,
                              lienF);
          probaTab[nomFonction] = tmpProba;
        }
        link[nomFonction] = lienF;
      }
    } else {
      std::cout << "instruction non prise en compre " << key << std::endl;
    }
  }

  return varIndex;
}

bool findCycle(std::map<std::size_t, std::vector<std::string>> const& varFonc,
               std::map<std::string, std::vector<std::size_t>>        link,
               std::string const&                                     cible,
               std::size_t                                            id,
               std::map<std::string, std::size_t>&                    vue)
{
  auto fonctionTmp = varFonc.at(id);
  bool parcour     = false;

  for (std::size_t i = 0; i < fonctionTmp.size() && !parcour; i++) {
    if (vue.find(fonctionTmp[i]) == vue.end()) {
      vue[fonctionTmp[i]] = 1;
      if (fonctionTmp[i] == cible) {
        parcour = true;
      }
      auto lien = link[fonctionTmp[i]];

      for (std::size_t j = 0; j < lien.size() && !parcour; j++) {
        if (lien[j] != id && varFonc.at(lien[j]).size() > 1) {
          auto tmp = varFonc;
          auto it  = tmp.begin();
          while (it != tmp.end()) {
            if (it->first == id) {
              auto erase = it;
              ++it;
              tmp.erase(erase);
            } else {
              ++it;
            }
          }
          parcour = findCycle(tmp, link, cible, lien[j], vue);
        }
      }
    }
  }

  return parcour;
}

bool findLittleCycle(std::map<std::size_t, std::vector<std::string>> const& varFonc,
                     std::string const&                                     cible,
                     std::size_t                                            id)
{
  std::vector<std::string> fonctionTmp = varFonc.at(id);
  bool                     parcour     = false;

  for (auto const& i : fonctionTmp) {
    if (i == cible) {
      parcour = true;
    }
  }

  return parcour;
}

std::vector<std::size_t> hammingToDecimal(std::size_t h)
{
  std::vector<std::size_t> value;
  for (std::size_t i = 0; i < 256; i++) {
    if (NumberOfSetBits(i) == h) {
      value.push_back(i);
    }
  }

  return value;
}

std::size_t NumberOfSetBits(std::size_t i)
{
  i = i - ((i >> 1) & 0x55555555);
  i = (i & 0x33333333) + ((i >> 2) & 0x33333333);

  return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

void hammingInstructionIteration(std::string const&                        key,
                                 std::size_t&                              ValeurKey,
                                 std::string&                              hammingweight,
                                 std::vector<std::string> const&           contenue,
                                 std::size_t                               i,
                                 std::map<std::string, std::size_t> const& instruction)
{
  if (instruction.at(key) > 4 && instruction.at(key) < 6) {
    if (std::string::npos != contenue[i].find("Z+") || std::string::npos != contenue[i].find("X+") ||
        std::string::npos != contenue[i].find("Y+")) {
      hammingweight     = hammingweight.substr(hammingweight.find(',') + 1);
      hammingweight     = hammingweight.substr(hammingweight.find(',') + 1);
      std::string tempo = hammingweight.substr(0, hammingweight.find(','));
      hammingweight     = hammingweight.substr(hammingweight.find(',') + 1);
      hammingweight     = hammingweight.substr(hammingweight.find(',') + 1);
      hammingweight     = hammingweight.substr(hammingweight.find(',') + 1);
      hammingweight     = tempo + "," + hammingweight;
      ValeurKey         = 1;
    } else {
      hammingweight     = hammingweight.substr(hammingweight.find(',') + 1);
      std::string tempo = hammingweight.substr(0, hammingweight.find(','));
      hammingweight     = hammingweight.substr(hammingweight.find(',') + 1);
      hammingweight     = hammingweight.substr(hammingweight.find(',') + 1);
      hammingweight     = tempo + "," + hammingweight;
      ValeurKey         = 1;
    }
  } else if (instruction.at(key) > 6 || key == "RET") {
    for (std::size_t compteur = 0; compteur < instruction.at(key); compteur++) {
      hammingweight = hammingweight.substr(hammingweight.find(',') + 1);
    }
    ValeurKey = 0;
  } else if (key == "SBIW" || key == "ADIW" || key == "MOVW") {
    hammingweight     = hammingweight.substr(hammingweight.find(',') + 1);
    hammingweight     = hammingweight.substr(hammingweight.find(',') + 1);
    std::string tempo = hammingweight.substr(0, hammingweight.find(','));
    hammingweight     = hammingweight.substr(hammingweight.find(',') + 1);
    hammingweight     = hammingweight.substr(hammingweight.find(',') + 1);
    hammingweight     = tempo + "," + hammingweight;
    ValeurKey         = 1;
  } else {
    hammingweight     = hammingweight.substr(hammingweight.find(',') + 1);
    std::string tempo = hammingweight.substr(0, hammingweight.find(','));
    hammingweight     = hammingweight.substr(hammingweight.find(',') + 1);
    hammingweight     = hammingweight.substr(hammingweight.find(',') + 1);
    hammingweight     = tempo + "," + hammingweight;
    ValeurKey         = 1;
  }
}

void instructionTaken(std::map<std::string, std::size_t>& instruction, std::map<std::string, std::size_t>& traitement)
{
  instruction["ADD"]   = 3;
  instruction["ADIW"]  = 4;
  instruction["ADC"]   = 3;
  instruction["LDI"]   = 1;
  instruction["OUT"]   = 2;
  instruction["EOR"]   = 3;
  instruction["CPI"]   = 1;
  instruction["CPC"]   = 2;
  instruction["LPM"]   = 5;
  instruction["MOV"]   = 2;
  instruction["MOVW"]  = 2;
  instruction["LDD"]   = 4;
  instruction["ANDI"]  = 2;
  instruction["SBCI"]  = 2;
  instruction["STD"]   = 4;
  instruction["RCALL"] = 34;
  instruction["CALL"]  = 34;
  instruction["SBSR"]  = 1;
  instruction["SBIW"]  = 4;
  instruction["DEC"]   = 2;
  instruction["ST"]    = 5;
  instruction["LD"]    = 5;
  instruction["RET"]   = 4;
  instruction["SUB"]   = 3;
  instruction["AND"]   = 3;

  traitement["ADD"] = 0;
  traitement["EOR"] = 1;
  traitement["SUB"] = 2;
  traitement["AND"] = 3;
}

void specialInstruction(std::string const&                            key,
                        std::map<std::string, std::size_t> const&     traitement,
                        std::vector<std::vector<std::vector<float>>>& tmpProba,
                        std::vector<std::vector<float>> const&        tabProbFonc,
                        std::size_t                                   nbLabel)
{
  if (traitement.find(key) != traitement.end()) {
    std::size_t nombreExis = 0;
    switch (traitement.at(key)) {
    case 0:
      for (std::size_t cpt = 0; cpt < nbLabel; cpt++) {
        std::vector<std::vector<float>> tmp(nbLabel);
        tmpProba[cpt] = tmp;
        for (std::size_t cpt2 = 0; cpt2 < nbLabel; cpt2++) {
          std::vector<float> tmp2(nbLabel);
          tmpProba[cpt][cpt2] = tmp2;
          for (std::size_t cpt3 = 0; cpt3 < nbLabel; cpt3++) {
            if (tabProbFonc[0][cpt] != 0.F && tabProbFonc[1][cpt2] != 0.F && tabProbFonc[2][cpt3] != 0.F) {
              if ((cpt + cpt2) == cpt3) {
                tmpProba[cpt][cpt2][cpt3] = 1;
                nombreExis++;
              }
            } else {
              tmpProba[cpt][cpt2][cpt3] = 0;
            }
          }
        }
      }
      for (std::size_t cpt = 0; cpt < nbLabel; cpt++) {
        for (std::size_t cpt2 = 0; cpt2 < nbLabel; cpt2++) {
          for (std::size_t cpt3 = 0; cpt3 < nbLabel; cpt3++) {
            if (tmpProba[cpt][cpt2][cpt3] == 1.F) {
              tmpProba[cpt][cpt2][cpt3] = 1.F / nombreExis;
            }
          }
        }
      }
      break;
    case 1:
      for (std::size_t cpt = 0; cpt < nbLabel; cpt++) {
        std::vector<std::vector<float>> tmp(nbLabel);
        tmpProba[cpt] = tmp;
        for (std::size_t cpt2 = 0; cpt2 < nbLabel; cpt2++) {
          std::vector<float> tmp2(nbLabel);
          tmpProba[cpt][cpt2] = tmp2;
          for (std::size_t cpt3 = 0; cpt3 < nbLabel; cpt3++) {
            if (tabProbFonc[0][cpt] != 0.F && tabProbFonc[1][cpt2] != 0.F && tabProbFonc[2][cpt3] != 0.F) {
              std::size_t xortmp = cpt;
              xortmp ^= cpt2;
              if (xortmp == cpt3) {
                tmpProba[cpt][cpt2][cpt3] = 1;
                nombreExis++;
              }
            } else {
              tmpProba[cpt][cpt2][cpt3] = 0;
            }
          }
        }
      }
      for (std::size_t cpt = 0; cpt < nbLabel; cpt++) {
        for (std::size_t cpt2 = 0; cpt2 < nbLabel; cpt2++) {
          for (std::size_t cpt3 = 0; cpt3 < nbLabel; cpt3++) {
            if (tmpProba[cpt][cpt2][cpt3] == 1.F) {
              tmpProba[cpt][cpt2][cpt3] = 1.F / nombreExis;
            }
          }
        }
      }
      break;
    case 2:
      for (std::size_t cpt = 0; cpt < nbLabel; cpt++) {
        std::vector<std::vector<float>> tmp(nbLabel);
        tmpProba[cpt] = tmp;
        for (std::size_t cpt2 = 0; cpt2 < nbLabel; cpt2++) {
          std::vector<float> tmp2(nbLabel);
          tmpProba[cpt][cpt2] = tmp2;
          for (std::size_t cpt3 = 0; cpt3 < nbLabel; cpt3++) {
            if (tabProbFonc[0][cpt] != 0.F && tabProbFonc[1][cpt2] != 0.F && tabProbFonc[2][cpt3] != 0.F) {
              if ((cpt - cpt2) == cpt3) {
                tmpProba[cpt][cpt2][cpt3] = 1;
                nombreExis++;
              }
            } else {
              tmpProba[cpt][cpt2][cpt3] = 0;
            }
          }
        }
      }
      for (std::size_t cpt = 0; cpt < nbLabel; cpt++) {
        for (std::size_t cpt2 = 0; cpt2 < nbLabel; cpt2++) {
          for (std::size_t cpt3 = 0; cpt3 < nbLabel; cpt3++) {
            if (tmpProba[cpt][cpt2][cpt3] == 1.F) {
              tmpProba[cpt][cpt2][cpt3] = 1.F / nombreExis;
            }
          }
        }
      }
      break;
    case 3:
      for (std::size_t cpt = 0; cpt < nbLabel; cpt++) {
        std::vector<std::vector<float>> tmp(nbLabel);
        tmpProba[cpt] = tmp;
        for (std::size_t cpt2 = 0; cpt2 < nbLabel; cpt2++) {
          std::vector<float> tmp2(nbLabel);
          tmpProba[cpt][cpt2] = tmp2;
          for (std::size_t cpt3 = 0; cpt3 < nbLabel; cpt3++) {
            if (tabProbFonc[0][cpt] != 0.F && tabProbFonc[1][cpt2] != 0.F && tabProbFonc[2][cpt3] != 0.F) {
              std::size_t andTmp = cpt;
              andTmp &= cpt2;
              if (andTmp == cpt3) {
                tmpProba[cpt][cpt2][cpt3] = 1;
                nombreExis++;
              }
            } else {
              tmpProba[cpt][cpt2][cpt3] = 0;
            }
          }
        }
      }
      for (std::size_t cpt = 0; cpt < nbLabel; cpt++) {
        for (std::size_t cpt2 = 0; cpt2 < nbLabel; cpt2++) {
          for (std::size_t cpt3 = 0; cpt3 < nbLabel; cpt3++) {
            if (tmpProba[cpt][cpt2][cpt3] == 1.F) {
              tmpProba[cpt][cpt2][cpt3] = 1.F / nombreExis;
            }
          }
        }
      }
      break;
    default:
      std::cout << "error isntruction \n";
    }
  } else {
    for (std::size_t cpt = 0; cpt < nbLabel; cpt++) {
      std::vector<std::vector<float>> tmp(nbLabel);
      tmpProba[cpt] = tmp;
      for (std::size_t cpt2 = 0; cpt2 < nbLabel; cpt2++) {
        std::vector<float> tmp2(nbLabel);
        tmpProba[cpt][cpt2] = tmp2;
        for (std::size_t cpt3 = 0; cpt3 < nbLabel; cpt3++) {
          tmpProba[cpt][cpt2][cpt3] = tabProbFonc[0][cpt] * tabProbFonc[1][cpt2] * tabProbFonc[2][cpt3];
        }
      }
    }
  }
}

void standartInstruction(std::string const&                                     key,
                         std::vector<std::vector<std::vector<float>>>&          tmpProba,
                         std::size_t                                            nbLabel,
                         std::map<std::string, std::size_t>&                    fonctions,
                         std::string const&                                     nomFonction,
                         std::vector<std::vector<float>> const&                 tabProbFonc,
                         std::vector<std::size_t>                               stockR30,
                         std::vector<std::size_t>                               stockR28,
                         std::vector<std::size_t>                               stockR26,
                         std::vector<std::size_t>                               stockLPMR30,
                         std::vector<std::size_t>                               stockLPMR28,
                         std::vector<std::size_t>                               stockLPMR26,
                         std::map<std::string, std::size_t> const&              varIndex,
                         std::map<std::size_t, std::vector<std::string>>&       varFonc,
                         std::size_t                                            i,
                         bool                                                   LPMReg,
                         std::size_t                                            idLPMReg,
                         std::string const&                                     ligneEnCours,
                         std::map<std::size_t, std::vector<std::size_t>> const& box,
                         std::vector<std::size_t>&                              lienF)
{
  std::vector<std::vector<float>> tmp(1);
  tmpProba[0] = tmp;

  std::vector<float> tmp2(nbLabel);
  tmpProba[0][0] = tmp2;
  if (key == "LPM" && box.find(i) != box.end()) {
    fonctions[nomFonction] = 2;
    std::size_t                     iden;
    std::vector<std::vector<float>> tmp3(nbLabel);
    tmpProba[0] = tmp3;
    for (std::size_t att = 0; att < nbLabel; att++) {
      std::vector<float> tmp4(nbLabel);
      tmpProba[0][att] = tmp4;
    }

    std::vector<std::size_t> offset;
    std::size_t              indexLigne = ligneEnCours.find(',');
    if (LPMReg) {
      iden = idLPMReg;
      if (ligneEnCours.substr(indexLigne, indexLigne + 2).find('Z') != std::string::npos) {
        offset = std::move(stockLPMR30);
      } else if (ligneEnCours.substr(indexLigne, indexLigne + 2).find('Y') != std::string::npos) {
        offset = std::move(stockLPMR28);
      } else {
        offset = std::move(stockLPMR26);
      }
    } else {
      if (ligneEnCours.substr(indexLigne, indexLigne + 2).find('Z') != std::string::npos) {
        offset = std::move(stockR30);
        iden   = varIndex.at("R30");
      } else if (ligneEnCours.substr(indexLigne, indexLigne + 2).find('Y') != std::string::npos) {
        offset = std::move(stockR28);
        iden   = varIndex.at("R28");
      } else {
        offset = std::move(stockR26);
        iden   = varIndex.at("R26");
      }
    }
    lienF.push_back(iden);
    varFonc[iden].push_back(nomFonction);
    // varIndex[nomVar]=iden;

    for (std::size_t v = 0; v < nbLabel; v++) {
      for (std::size_t a : offset) {
        if (a < box.at(i).size()) {
          std::size_t tmp = box.at(i)[a];
          if (tabProbFonc[0][v] > 0 && tmp == v) {
            tmpProba[0][v][a] = 1;
          }
        }
      }
    }
  } else {
    for (std::size_t cpt3 = 0; cpt3 < nbLabel; cpt3++) {
      tmpProba[0][0][cpt3] = tabProbFonc[0][cpt3];
    }
  }
}
