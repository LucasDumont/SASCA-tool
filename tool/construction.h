#ifndef CONSTRUCTION_H
#define CONSTRUCTION_H

#include <boost/lexical_cast.hpp>
#include <cstdlib>
#include <iostream>
#include <map>
#include <opengm/functions/potts.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/maximizer.hxx>
#include <string>

using Model = opengm::GraphicalModel<float, opengm::Adder>;

/**
 * @brief modelCreation this fonction a factor graph
 * @param numberOfVar number of variable in the model
 * @param fonction the functions which composed the model
 * @param link the link between functions and variables
 * @param vars the variable in the model
 * @return
 */
Model modelCreation(std::map<std::string, std::size_t> const&                            fonctions,
                    std::map<std::string, std::vector<std::vector<std::vector<float>>>>& probaTab,
                    std::map<std::string, std::vector<std::size_t>> const&               link,
                    std::vector<std::size_t> const&                                      var);
/**
 * @brief fonctionPremier
 * @param gm
 * @param prob
 * @param var
 */
void fonctionPremier(Model& gm, std::vector<float> const& prob, std::size_t var);
/**
 * @brief fonctionSecond
 * @param gm
 * @param prob
 * @param var
 */
void fonctionSecond(Model& gm, std::vector<std::vector<float>> const& prob, std::array<std::size_t, 2> const& var);
/**
 * @brief fonctionTroisieme
 * @param gm
 * @param prob
 * @param var
 */
void fonctionTroisieme(Model&                                              gm,
                       std::vector<std::vector<std::vector<float>>> const& prob,
                       std::array<std::size_t, 3> const&                   var);

/**
 * @brief beliefPropagation this fonction resolve a factor graph with belief
 * propagation
 * @param gm the model
 * @param output  the final output with all value per variable
 * @param iteration  the number of iteration for the belief propagation
 * @param allVariables true for have all varaible , false for prstd::size_t only the
 * most
 */
void beliefPropagation(Model const& gm, std::vector<std::string>& output, std::size_t iteration, bool allVariables);
/**
 * @brief transformationASM
 * @param contenue the vector whiwh contain the execution stack
 * @param probaTab the propablity table per function
 * @param link link beteween function and varaibel
 * @param var  variables in the model
 * @param fonctions function in the model
 * @param hammingweight the vector which contain the leakage
 * @param valeurFixer  the vector which contain clear text
 * @param valeurResultat  the vector which contain the cypher text
 * @param box the vector wich contain the sbox
 * @param graph  true for juste generate a graph without probability
 * @param cycle true for generate a graph with cycle
 * @return
 */
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
                  bool                                                                 cycle);
/**
 * @brief findCycle
 * @param varFonc
 * @param link
 * @param cible
 * @param id
 * @param vue
 * @return
 */
bool findCycle(std::map<std::size_t, std::vector<std::string>> const& varFonc,
               std::map<std::string, std::vector<std::size_t>> const& link,
               std::string const&                                     cible,
               std::size_t                                            id,
               std::map<std::string, std::size_t>&                    vue);
/**
 * @brief hammingToDecimal
 * @param h
 * @return
 */
std::vector<std::size_t> hammingToDecimal(std::size_t h);
/**
 * @brief NumberOfSetBits
 * @param i
 * @return
 */
std::size_t NumberOfSetBits(std::size_t i);
/**
 * @brief findLittleCycle
 * @param varFonc
 * @param cible
 * @param id
 * @return
 */
bool findLittleCycle(std::map<std::size_t, std::vector<std::string>> const& varFonc,
                     std::string const&                                     cible,
                     std::size_t                                            id);
/**
 * @brief hammingInstructionIteration
 * @param key
 * @param ValeurKey
 * @param hammingweight
 * @param contenue
 * @param i
 * @param instruction
 */
void hammingInstructionIteration(std::string const&                        key,
                                 std::size_t&                              ValeurKey,
                                 std::string&                              hammingweight,
                                 std::vector<std::string> const&           contenue,
                                 std::size_t                               i,
                                 std::map<std::string, std::size_t> const& instruction);
/**
 * @brief instructionTaken
 * @param instruction
 * @param traitement
 */
void instructionTaken(std::map<std::string, std::size_t>& instruction, std::map<std::string, std::size_t>& traitement);
/**
 * @brief specialInstruction
 * @param key
 * @param traitement
 * @param tmpProba
 * @param tabProbFonc
 * @param nbLabel
 */
void specialInstruction(std::string const&                            key,
                        std::map<std::string, std::size_t> const&     traitement,
                        std::vector<std::vector<std::vector<float>>>& tmpProba,
                        std::vector<std::vector<float>> const&        tabProbFonc,
                        std::size_t                                   nbLabel);
/**
 * @brief standartInstruction
 * @param key
 * @param tmpProba
 * @param nbLabel
 * @param fonctions
 * @param nomFonction
 * @param tabProbFonc
 * @param stockR30
 * @param stockR28
 * @param stockR26
 * @param stockLPMR30
 * @param stockLPMR28
 * @param stockLPMR26
 * @param varIndex
 * @param varFonc
 * @param i
 * @param LPMReg
 * @param idLPMReg
 * @param ligneEnCours
 * @param box
 * @param lienF
 */
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
                         std::map<std::string, std::size_t>                     varIndex,
                         std::map<std::size_t, std::vector<std::string>>&       varFonc,
                         std::size_t                                            i,
                         bool                                                   LPMReg,
                         std::size_t                                            idLPMReg,
                         std::string const&                                     ligneEnCours,
                         std::map<std::size_t, std::vector<std::size_t>> const& box,
                         std::vector<std::size_t>&                              lienF);

#endif // CONSTRUCTION_H
