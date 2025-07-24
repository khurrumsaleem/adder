from collections import OrderedDict
import numpy
import os
import copy
import re
import h5py
import numpy as np
import pyparsing as pp

from adder.type_checker import *
from .constants import TALLY_CARDS, TMESH_CARDS, ADDER_TALLY_ID, ADDER_USER_TALLY_ID, \
    TALLY_MAX_ID, VALID_FCARD_TALLY_TYPES, VALID_TALLY_TYPES, PARTICLE_TYPES
from adder.loggedclass import LoggedClass
from adder.utils import get_id

logger = LoggedClass(0, __name__)
_INDENT = 2

"""
List of functions used to process the tally class during
fuel management operations.
"""


def map_entities(model_entities, attr="parent_id"):
    """Function to map entities, i.e. child to parent ids cells, cell to material ids, etc."""
    # get parents ids list
    model_entit_parents_ids = list(map(lambda id: str(getattr(model_entities[id], attr)), model_entities.keys()))

    # get ids in string form & link mapping
    model_entit_ids = [str(cell_id) for cell_id in model_entities.keys()]
    model_entit_map = dict(zip(model_entit_ids, model_entit_parents_ids))

    return model_entit_map


def card_entities_parsing(card_entities):
    """parsing card string specification into list of entity groups"""

    # defining matching expressions to parse tally card specification.
    integer = pp.Word(pp.nums).setResultsName("integer")
    nested_expr = pp.nestedExpr("(", ")").setResultsName("composite")
    nested_expr2 = pp.nestedExpr("[", "]").setResultsName("composite2")

    # defining general expression to parse tally card specification
    # into a list w/ nested lists corresponding to groups of entities.
    expr = integer | nested_expr | nested_expr2
    card_parsed = expr.searchString(card_entities)

    # creating list w/ nested lists for groups of entities.
    entit_groups_l = [reconstruct(item) for item in card_parsed]

    return entit_groups_l


def reconstruct(item):
    """helper functions used by card_entities_parsing"""

    if isinstance(item, str):
        return item
    elif isinstance(item, list):
        return '(' + ' '.join([reconstruct(x) for x in item]) + ')'
    elif item.get_name() == 'integer':
        return str(item[0])
    elif item.get_name() == 'composite':
        return '(' + ' '.join([reconstruct(x) for x in item[0].asList()]) + ')'
    elif item.get_name() == 'composite2':
        return '[' + ' '.join([reconstruct(x) for x in item[0].asList()]) + ']'


def match_tally_id_entity(id_cell_tally, model_cells_map,
                          clone_entit_groups_d):
    """
    Function for tally match and copy w/ entity group simple form definition ,
    i.e. C1, ..., Cn.
    This function create/update clone_entit_groups_d, i.e. dictionary w/
    each entry corresponding to f card string of each tally clone
    """
    idx_copy = 0
    # find match cell_ids and parents
    tally_match, tally_copy = False, False
    for model_cell_id, parent_id in model_cells_map.items():
        if id_cell_tally == model_cell_id:
            tally_match = True
        elif id_cell_tally == parent_id:
            idx_copy = idx_copy + 1
            if idx_copy not in clone_entit_groups_d.keys():
                clone_entit_groups_d[idx_copy] = str(model_cell_id)
            else:
                clone_entit_groups_d[idx_copy] = clone_entit_groups_d[idx_copy] + str(" " + str(model_cell_id))
            tally_copy = True
    return tally_match, tally_copy, clone_entit_groups_d


def analyze_complex_entit_group(entit_group, model_cells_map, parent_map, entity_type_f_card):
    """Dedicated function to analyze complex entity groups"""

    # remove space between parhenteses
    entit_group = re.sub(r'\s*([()])\s*', r'\1', entit_group)
    entit_group_parsed = parse_complex_entity_group(entit_group, parhenteses=False)
    clone_entit_group_parsed = {}

    # check if tally itself match for all cells in the geometry
    tally_match = True
    for i, sublist in enumerate(entit_group_parsed):
        for j, element in enumerate(sublist):
            # no need check if surf tally at 1st sublist (see MCNP manual)
            if (entity_type_f_card == "cell") or (entity_type_f_card == "surface" and i > 0):
                # check correspondence
                if entit_group_parsed[i][j].startswith("["):
                    pass
                elif (entit_group_parsed[i][j] not in model_cells_map.keys()):
                    tally_match = False
                    break

    # check match for tally copy
    tally_copy = True
    for i, sublist in enumerate(entit_group_parsed):
        for j, element in enumerate(sublist):

            # no need check if surf tally at 1st sublist (see MCNP manual)
            if (entity_type_f_card == "cell") or (entity_type_f_card == "surface" and i > 0):

                # check correspondence
                if (entit_group_parsed[i][j] in model_cells_map.keys()) or (
                        entit_group_parsed[i][j] in model_cells_map.values()):
                    # check combination
                    for cell_id, parent_id in model_cells_map.items():

                        # create or update clone_entity_group_parsed dictionary
                        if entit_group_parsed[i][j] == parent_id:
                            # add complex entity copy
                            clone_entit_group_parsed = check_complex_entity_tocopy(clone_entit_group_parsed, parent_id,
                                                                                   parent_map, entit_group_parsed, i, j)
                elif entit_group_parsed[i][j].startswith("["):
                    pass
                else:
                    tally_copy = False
                    break

    return tally_match, tally_copy, clone_entit_group_parsed


def parse_complex_entity_group(entit_group, parhenteses=True):
    """
    parse complex entity to get list w/ sublists containing the parentheses
    (if present) for complex entity:
              ((Si..Sn)<(Cm Cn[Ij..Ik])<(Cl Ck[Ij..Ik])...(Cp Co))
    The parsed complex entity results:
             [[Si,..,Sn], [Cm, Cn, "[Ij...Ik]"], [Cl, Ck, "[Ij..Ik]"]..]
    """

    entit_group_split = entit_group[1:-1].split("<")

    entit_group_parsed = []

    # parse complex entity to a list w/ sublist
    for entit in entit_group_split:

        # distinguish between case within parentheses and without
        if entit.startswith('(') and entit.endswith(')') and not parhenteses:
            entit_parse = entit[1:-1]
        else:
            entit_parse = entit

        sublist_entity_group = card_entities_parsing(entit_parse)
        entit_group_parsed.append(sublist_entity_group)

    return entit_group_parsed


def check_complex_entity_tocopy(clone_entit_group_parsed, parent_id, parent_map, entit_group_parsed, i, j):
    """
    update new_entit_group_parsed dictionary for clone complex entity
    each key of new_entit_group_parsed is a complex entity of a clone
    clone_entit_group_parsed
         {"1": [[Si,..,Sn], [Cm, Cn, "[Ij...Ik]"], [Cl, Ck, "[Ij..Ik]"]..]
          "2": [[Si,..,Sn], [Cm, Cn, "[Ij...Ik]"], [Cl, Ck, "[Ij..Ik]"]..]
          ...
          "n": [[Si,..,Sn], [Cm, Cn, "[Ij...Ik]"], [Cl, Ck, "[Ij..Ik]"]..]}
     where each key is associated to a list w/ clone cell ids
    """

    # numbers of copies
    n_copy = len(parent_map[parent_id])
    n_parsed = len(clone_entit_group_parsed)

    # check consistency number of copies w/ previous
    # parent ids analyzed
    if clone_entit_group_parsed and n_parsed != n_copy:
        # when multiple copies are created from a single universes filling
        # a lattice. Recommended to define at the beginning a univ for each lattice elem.
        if n_parsed > n_copy:
            for idx_parsed in range(1, n_parsed + 1):
                for idx_copy in range(1, n_copy + 1):
                    clone_entit_group_parsed[idx_parsed][i][j] = parent_map[parent_id][idx_copy - 1]

    # create/updated clone_entit_group_parsed dictionary
    for idx_copy in range(1, n_copy + 1):
        if idx_copy not in clone_entit_group_parsed.keys():
            clone_entit_group_parsed[idx_copy] = copy.deepcopy(entit_group_parsed)
            clone_entit_group_parsed[idx_copy][i][j] = parent_map[parent_id][idx_copy - 1]
        else:
            clone_entit_group_parsed[idx_copy][i][j] = parent_map[parent_id][idx_copy - 1]
    return clone_entit_group_parsed


def reconstr_entit_group_parsed(entit_group_parsed, entit_group_ref):
    """
    Reconstructing the entity group parsed (list) to the corresponding
    string, maintaining the original entity group string structure.
    For the string structure, a reference entity group string is
    provided.
    """

    # Format sublist
    entit_group_ref_parsed = parse_complex_entity_group(entit_group_ref)
    formatted_sublists = []
    for i in range(len(entit_group_parsed)):
        if entit_group_ref_parsed[i][0].startswith("(") and entit_group_ref_parsed[i][0].endswith(")"):
            formatted_sublists.append("(" + " ".join(entit_group_parsed[i]) + ")")
        else:
            formatted_sublists.append(" ".join(entit_group_parsed[i]))
    # Construct complete string
    clone_entit_group_s = "(" + " < ".join(formatted_sublists) + ")"

    return clone_entit_group_s


def substit_f_card(f_card, map_dict):
    # Regular expression to match numbers not within square brackets
    pattern = r'(?<!\[)(?<!\d)(\b\d+\b)(?!\d)(?!\])'

    # Function to replace matched numbers with their corresponding values
    def replace(match):
        # Extract the matched number as an integer
        num = int(match.group(1))
        # Replace the number with its corresponding value from the dictionary, or keep the number if not in the dictionary
        return str(map_dict.get(num, num))

    # Substitute the found numbers with their corresponding values
    new_f_card = re.sub(pattern, replace, f_card)

    return new_f_card


def inverse_map_cells(original_map):
    """Create a new dictionary by exchanging keys and values"""

    inv_map = {value: key for key, value in original_map.items()}
    return inv_map


def convert_map_to_int(map_dict):
    """function to convert string key to int"""

    int_map_dict = {}
    for key, value in map_dict.items():
        try:
            int_key = int(key)
            int_value = int(value)
            int_map_dict[int_key] = int_value
        except ValueError:
            raise ValueError(f"Both key and value must be convertible to int. Found: key={key}, value={value}")
    return int_map_dict


def fmstring_to_nested_list(fm_string):
    """get nested lists for fm card"""

    nested_list = []
    stack = [nested_list]

    # Remove "fm" prefix and split the string into tokens
    tokens = fm_string.replace("(", " ( ").replace(")", " ) ").split()

    for token in tokens:
        if token == "(":
            new_list = []
            stack[-1].append(new_list)
            stack.append(new_list)
        elif token == ")":
            stack.pop()
        else:
            stack[-1].append(token)

    return nested_list


def nested_list_to_fmstring(nested_list):
    """reconstructes fm card string from nested lists"""

    def recurse(sublist):
        if isinstance(sublist, list):
            return '(' + ' '.join(recurse(item) for item in sublist) + ')'
        else:
            return str(sublist)

    return recurse(nested_list)[1:-1]


def subst_id_mat_fm(nested, material_map):
    """substitutes material id for attenuator and multiplier set"""

    if len(nested) > 2:

        # attenuator set
        if isinstance(nested[1], str) and nested[1] == "-1":
            n_mats_att = int((len(nested) - 2) / 2)
            for j in range(n_mats_att):
                idx = 2 * (1 + j)

                if nested[idx] in material_map:
                    nested[idx] = material_map[nested[idx]]

        # multiplier set
        elif isinstance(nested[1], str) and nested[1] in material_map:
            nested[1] = material_map[nested[1]]

    return nested


def check_nested_l(nested_l, material_map):
    """check id material in fm card and substitute with material mapped """
    # substitution id
    nested_l = subst_id_mat_fm(nested_l, material_map)

    # find inner lists to be checked
    inner_l_ids = []
    for index, element in enumerate(nested_l):
        if isinstance(element, list):
            inner_l_ids.append(index)

    # inner iterations to susbstitute mat ids.
    if inner_l_ids != []:
        for idx in inner_l_ids:
            nested_l[idx] = check_nested_l(nested_l[idx], material_map)

    return nested_l


def check_tally_ids(ids_tally_mcnp, ids_tally_adder):
    """filter tally ids with the ones set in the ADDER input"""
    ids_tally_check = []
    for tally_id in ids_tally_adder:
        tally_id = int(tally_id)
        if tally_id not in ids_tally_mcnp:
            logger.log("info_file", f"Tally {tally_id} in ADDER input not"
                                    f" found in initial MCNP input. Not processed.", 10)
        else:
            ids_tally_check.append(tally_id)

    return ids_tally_check


def update_tally_dict(id, card_type, card_s, tally_dict):
    """update tally dict. by creating or updating a new tally """
    if id not in tally_dict.keys():
        tally_dict[id] = Tally(id=id)
    # distinguish between other_cards and the rest of the cards
    if card_type != "other_cards":
        setattr(tally_dict[id], card_type, card_s)
        tally_dict[id].extract_related_characteristics()
    else:
        if tally_dict[id].other_cards:
            tally_dict[id].other_cards.append(card_s)
        else:
            tally_dict[id].other_cards = [card_s]

    return tally_dict


class Tally:
    """An MCNP Tally object.

        Parameters
        ----------
        id : int
            The tally's ID, corresponding  n in the card name Fn, FMn,
            FCn, etc.
        f_card: str
            The tally specification of the Fn card
        f_card_mat : str
            It corresponds to a string having the same structure of f_card,
            but the cells ids in the f_card are substituted by the material
            ids filling those cells.
        fc_card : str
            The tally specification of the FCn card
        fm_card : str
            The tally specification of the FMn card
        cf_card : str
            The tally specification of the CFn card
        other_cards : list
            list of the lines (str) for other tally card types, not
            specifically processed by ADDER (En, Tn, FQn, DEn, DFn,
            EMn, TMn, Fun/TALLYX, Fn, DDn, DXT, FTn, SPDTL)
        id_type : int
            The identifier for the tally types as reported in the MCNP manual
            (1 for current integrated over a surf., 2 for flux averaged over a
            surf., 4 for flux averaged over a cell, etc.), where the tally id
            is defined as this identifier or increments of 10 thereof.
        tally_block: list
            list of the lines (str) for each tally card specification.
        entity_type_f_card : str
            The entities, "surf" or "cell", linked to the tally through
            their ids, depending on id_type
        type: str
            it corresponds to "universe" or "material", referring if ADDER
            processes universes or materials during fuel management operations.
        particles: list
            The particle list for the particles tracked by the tally.
        material_names: list
            List of the material names linked to tally.
        universe_names: list
            List of the universe names linked to tally.
        facet_ids: list
            List of ids for the facets linked to tally.
        tally_matrix: arr
            The tally matrix corresponding to the 9-dimensional array for the
             tally results
        tally_matrix_err: arr
            it corresponds to the 9-dimensional array of the error associated
            to the tally matrix
        Attributes
        ----------
         id : int
            The tally's ID, corresponding  n in the card name Fn, FMn,
            FCn, etc.
        f_card: str
            The tally specification of the Fn card
        f_card_mat : str
            It corresponds to a string having the same structure of f_card,
            but the cells ids in the f_card are substituted by the material
            ids filling those cells.
        fc_card : str
            The tally specification of the FCn card
        fm_card : str
            The tally specification of the FMn card
        cf_card : str
            The tally specification of the CFn card
        other_cards : list
            list of the lines (str) for other tally card types, not
            specifically processed by ADDER
            (En, Tn, FQn, DEn, DFn, EMn, TMn, Fun/TALLYX, Fn, DDn, DXT, FTn,
            SPDTL)
        id_type : int
            The identifier for the tally types as reported in the MCNP manual
            (1 for current integrated over a surf., 2 for flux averaged over
            a surf., 4 for flux averaged over a cell, etc.), where the tally
            id is defined as this identifier or increments of 10 thereof.
        tally_block: list
            list of the lines (str) for each tally card specification.
        entity_type_f_card : str
            The entities ("facets"), "surf" or "cell", linked to the tally
            through their ids, depending on id_type
        type: str
            it corresponds to "universe" or "material", referring if ADDER
            processes universes or materials during fuel management operations.
        particles: list
            The particle list for the particles tracked by the tally.
        material_names: list
            List of the material names linked to tally.
        universe_names: list
            List of the universe names linked to tally.
        facet_ids: list
            List of ids for the facets linked to tally.
        tally_matrix: arr
            The tally matrix corresponding to the 9-dimensional array
            for the tally results
        tally_matrix_err: arr
            it corresponds to the 9-dimensional array of the error associated
            to the tally matrix
        """

    _USED_IDS = set([])

    def __init__(self, id, f_card=None, f_card_mat=None, fc_card=None,
                 fm_card=None, cf_card=None, fmesh_card=None, other_cards=[],
                 tally_block=[], entity_type_f_card=None, type=None,
                 particles=["n"], material_names=[], universe_names=[],
                 facet_ids=[], tally_matrix=np.array([]),
                 tally_matrix_err=np.array([])):
        self.id = id
        self.f_card = f_card
        self.f_card_mat = f_card_mat
        self.fc_card = fc_card
        self.fm_card = fm_card
        self.cf_card = cf_card
        self.fmesh_card = fmesh_card
        self.other_cards = other_cards
        self.tally_block = tally_block
        self.id_type = id % 10
        self.entity_type_f_card = entity_type_f_card
        self.type = type
        self.particles = particles
        self.universe_names = universe_names
        self.material_names = material_names
        self.facet_ids = facet_ids
        self.tally_matrix = tally_matrix
        self.tally_matrix_err = tally_matrix_err
        self.extract_related_characteristics()

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id_):
        if id_ is None:
            # Then we have to auto-generate an ID, use our generalized function
            self._id = get_id(Tally._USED_IDS, TALLY_MAX_ID)
        else:
            check_type("id", id_, int, None)
            check_greater_than("id", id_, 0, equality=True)
            check_less_than("id", id_, TALLY_MAX_ID, equality=True)
            self._id = id_
            Tally._USED_IDS.add(self._id)

    @property
    def f_card(self):
        return self._f_card

    @f_card.setter
    def f_card(self, f_card):
        if f_card is not None:
            check_type("f_card", f_card, str)
        self._f_card = f_card

    @property
    def f_card_mat(self):
        return self._f_card_mat

    @f_card_mat.setter
    def f_card_mat(self, f_card_mat):
        if f_card_mat is not None:
            check_type("f_card_mat", f_card_mat, str)
        self._f_card_mat = f_card_mat

    @property
    def fc_card(self):
        return self._fc_card

    @fc_card.setter
    def fc_card(self, fc_card):
        if fc_card is not None:
            check_type("fc_card", fc_card, str)
        self._fc_card = fc_card

    @property
    def fm_card(self):
        return self._fm_card

    @fm_card.setter
    def fm_card(self, fm_card):
        if fm_card is not None:
            check_type("fm_card", fm_card, str)
        self._fm_card = fm_card

    @property
    def cf_card(self):
        return self._cf_card

    @cf_card.setter
    def cf_card(self, cf_card):
        if cf_card is not None:
            check_type("cf_card", cf_card, str)
        self._cf_card = cf_card

    @property
    def fmesh_card(self):
        return self._fmesh_card

    @fmesh_card.setter
    def fmesh_card(self, fmesh_card):
        if fmesh_card is not None:
            check_type("fmesh_card", fmesh_card, str)
        self._fmesh_card = fmesh_card

    @property
    def other_cards(self):
        return self._other_cards

    @other_cards.setter
    def other_cards(self, other_cards):
        if other_cards is not None:
            check_type("other_cards", other_cards, list)
        self._other_cards = other_cards

    @property
    def tally_block(self):
        return self._tally_block

    @tally_block.setter
    def tally_block(self, tally_block):
        if tally_block is not None:
            check_type("tally_block", tally_block, list)
        self._tally_block = tally_block

    @property
    def entity_type_f_card(self):
        return self._entity_type_f_card

    @entity_type_f_card.setter
    def entity_type_f_card(self, entity_type_f_card):
        if entity_type_f_card is not None:
            check_type("entity_type_f_card", entity_type_f_card, str)
            check_value("entity_type_f_card", entity_type_f_card,
                        VALID_FCARD_TALLY_TYPES)
        self._entity_type_f_card = entity_type_f_card

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type):
        if type is not None:
            check_type("type", type, str)
            check_value("type", type, VALID_TALLY_TYPES)
        self._type = type

    @property
    def particles(self):
        return self._particles

    @particles.setter
    def particles(self, particles):
        check_type("particles", particles, list)
        for particle in particles:
            check_value("particle", particle, PARTICLE_TYPES)
        self._particles = particles

    @property
    def universe_names(self):
        return self._universe_names

    @universe_names.setter
    def universe_names(self, universe_names):
        check_type("universe_names", universe_names, list)
        self._universe_names = universe_names

    @property
    def material_names(self):
        return self._material_names

    @material_names.setter
    def material_names(self, material_names):
        check_type("material_names", material_names, list)
        self._material_names = material_names

    @property
    def facet_ids(self):
        return self._facet_ids

    @facet_ids.setter
    def facet_ids(self, facet_ids):
        check_type("facet_ids", facet_ids, list)
        self._facet_ids = facet_ids

    @property
    def tally_matrix(self):
        return self._tally_matrix

    @tally_matrix.setter
    def tally_matrix(self, tally_matrix):
        if tally_matrix is not None:
            check_type("tally_matrix", tally_matrix, np.ndarray)
        self._tally_matrix = tally_matrix

    @property
    def tally_matrix_err(self):
        return self._tally_matrix_err

    @tally_matrix_err.setter
    def tally_matrix_err(self, tally_matrix_err):
        if tally_matrix_err is not None:
            check_type("tally_matrix_err", tally_matrix_err, np.ndarray)
        self._tally_matrix_err = tally_matrix_err

    def extract_related_characteristics(self):
        """ extract other attributes from initialization """
        # extract cells/surfaces from f_card specification
        if self.f_card is not None:
            if self.id_type == 4 or self.id_type == 6 or self.id_type == 7:
                self.entity_type_f_card = "cell"
            elif self.id_type == 1 or self.id_type == 2:
                self.entity_type_f_card = "surface"

    def get_tally_block(self):
        """get list of tally cards"""

        tally_block = []
        card_entries_dict = {"fc": self.fc_card, "f": self.f_card,
                             "fm": self.fm_card, "cf": self.cf_card,
                             "fmesh": self.fmesh_card,
                             "other_cards": self.other_cards}

        # among tally cards get append a tally line to the tally block
        for card_name in card_entries_dict.keys():
            # different line for f card
            if card_name == "fc" and card_entries_dict[card_name]:
                tally_line = card_entries_dict[card_name]
                tally_block.append(tally_line)
            elif card_name == "f" and card_entries_dict[card_name]:
                for i in range(len(self.particles)):
                    if i == 0:
                        tally_line = card_name + str(self.id) + ":" \
                                     + self.particles[i]
                    else:
                        tally_line = tally_line + "," + self.particles[i]
                tally_line = tally_line + " " \
                             + str(card_entries_dict[card_name])
                tally_block.append(tally_line)
            elif card_name == "fm" and card_entries_dict[card_name]:
                tally_line = card_name + str(self.id) + " " \
                             + str(card_entries_dict[card_name])
                tally_block.append(tally_line)
            elif card_name == "cf" and card_entries_dict[card_name]:
                tally_line = str(card_entries_dict[card_name])
                tally_block.append(tally_line)
            elif card_name == "fmesh" and card_entries_dict[card_name]:
                # no management required for fmesh card
                tally_line = str(card_entries_dict[card_name])
                tally_block.append(tally_line)
            elif card_name == "other_cards" and card_entries_dict[card_name]:
                for line in card_entries_dict[card_name]:
                    tally_line = str(line)
                    tally_block.append(tally_line)
        return tally_block

    def clone(self, id_clone):

        # cloning tally and assigning tally cards to be processed by ADDER.
        clone = Tally(id_clone)
        clone.f_card = self.f_card
        clone.fc_card = self.fc_card
        clone.fm_card = self.fm_card
        clone.cf_card = self.cf_card
        clone.fmesh_card = self.fmesh_card
        clone.entity_type_f_card = self.entity_type_f_card
        clone.type = self.type
        clone.particles = self.particles
        # substitution of id for cards including the card identifiers (FMESHn, CFn, FCn)
        if self.fc_card:
            clone.fc_card = self.fc_card.replace(str(self.id), str(id_clone), 1)
        if self.cf_card:
            clone.cf_card = self.cf_card.replace(str(self.id), str(id_clone), 1)
        if self.fmesh_card:
            clone.fmesh_card = self.fmesh_card.replace(str(self.id), str(id_clone), 1)
        # assigning tally cards not to be processed by ADDER.
        if self.other_cards:
            clone.other_cards = []
            for card in self.other_cards:
                clone.other_cards.append(card.replace(str(self.id), str(id_clone), 1))

        return clone

    def to_hdf5(self, group):
        """Writes the tally to an opened HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to
        """

        # Now create data set to write. The HDF5 data structure produced by
        # this is based on minimizing the number of writes to the file which
        # helps locally (based on profile) and on networked drives.
        # To do this we have created a struct of the data in the tally obj.

        # Now the tally data
        dt_string = h5py.special_dtype(vlen=str)
        if not self.type:
            self.type = "unprocessed"
        if not self.entity_type_f_card:
            self.entity_type_f_card = "no facet type assigned"
        if not self.get_tally_block() or self.get_tally_block() == [] or self.tally_block == []:
            self.tally_block = ["not present in MCNP input"]
        else:
            self.tally_block = self.get_tally_block()
        if self.facet_ids == []:
            self.facet_ids = [0]
        if self.material_names == []:
            self.material_names = ["no materials"]
        if self.universe_names == []:
            self.universe_names = ["no universes"]
        if not self.tally_matrix.any():
            self.tally_matrix = np.zeros((1, 1, 1, 1, 1, 1, 1, 1, 1))
        if not self.tally_matrix_err.any():
            self.tally_matrix_err = np.zeros((1, 1, 1, 1, 1, 1, 1, 1, 1))

        struct = [
            ('id', np.int32),
            ('type', dt_string),
            ('facet_type', dt_string),
            ('input_specification', dt_string, (len(self.tally_block),)),
            ('material_names', dt_string, (len(self.material_names),)),
            ('universe_names', dt_string, (len(self.universe_names),)),
            ('facet_ids', np.int32, (len(self.facet_ids),)),
            ('tally_matrix', np.float64, np.shape(self.tally_matrix)),
            ('tally_matrix_err', np.float64, np.shape(self.tally_matrix_err))]

        # Now assign these values
        vals = \
            [(self.id, self.type, self.entity_type_f_card, self.tally_block, self.material_names,
              self.universe_names, self.facet_ids, self.tally_matrix, self.tally_matrix_err)]
        data = np.array(vals, dtype=np.dtype(struct))

        # here creation of data set
        dset = group.create_dataset(str(self.id), data=data)

    @classmethod
    def from_hdf5(cls, group, id):
        """Initializes a tally object from an opened HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from

        Returns
        -------
        this : Tally
            A Tally object initialized from the HDF5 file

        """

        # Lets get the dataset from the group
        dset = group[str(id)]
        data = dset[()]

        # Now extract the data we need
        id_ = int(data['id'][0])
        type_ = data['type'][0].decode('utf-8')
        facet_type_ = data['facet_type'][0].decode('utf-8')
        input_specification_ = [s.decode() for s in data['input_specification'][0]]
        universe_names_ = [s.decode() for s in data['universe_names'][0]]
        material_names_ = [s.decode() for s in data['material_names'][0]]
        facet_ids_ = list(data['facet_ids'][0])
        tally_matrix_ = data['tally_matrix'][0]
        tally_matrix_err_ = data['tally_matrix_err'][0]

        # Now we can create the isotopes
        this = cls(id=id_, tally_block=input_specification_, type=type_,
                   entity_type_f_card=facet_type_, particles=["n"],
                   material_names=material_names_,
                   universe_names=universe_names_, facet_ids=facet_ids_,
                   tally_matrix=tally_matrix_, tally_matrix_err=tally_matrix_err_)

        return this

    # List of functions used as utility to manage tallies

    def create_tally_clone(self, new_id, ids_list, entit_group, user_tallies_l):
        """
        check if entity group is present in user_tallies_l.
        if not, a new tally clone with those entity group is created
        """
        while new_id in ids_list:
            new_id += 10
        tally_clone = None
        f_card_l = [user_tally.f_card for user_tally in user_tallies_l]

        # check if f card different
        if entit_group not in f_card_l:
            tally_clone = self.clone(id_clone=new_id)
            tally_clone.f_card = entit_group
        return new_id, tally_clone

    def create_tally_clone_mat(self, new_id, ids_list, entit_group, user_tallies_l,
                               mats_ids, mats_ids_set):
        """create tally clone for materials"""
        while new_id in ids_list:
            new_id += 10

        # get the list of f_card_mat for user tallies
        tally_clone = None
        f_card_mat_l = [user_tally.f_card_mat for user_tally in user_tallies_l]

        # check if materials were already moved in previous operations
        if mats_ids.issubset(mats_ids_set):
            tally_match, tally_copy = False, False
            return tally_match, tally_copy, tally_clone, ids_list

        # check if f card mat not in other user tallies.
        # if not, clone creation.
        if entit_group not in f_card_mat_l:
            ids_list.append(new_id)
            tally_clone = self.clone(id_clone=new_id)

            # automatically, it clones all the others card associated.
            tally_clone.f_card_mat = entit_group

            # in the case of tally mat, there's no match to create
            # a copy. The copy is just created from the original cells
            # assigned
            tally_match, tally_copy = False, True

            return tally_match, tally_copy, tally_clone, ids_list
        else:
            tally_match, tally_copy = False, False

            return tally_match, tally_copy, tally_clone, ids_list

    def create_tally_clone_l(self, entit_groups_d, user_tallies_l, ids_list):
        """create tally clone list."""
        tally_clones_l = []
        for entit_group in entit_groups_d.values():
            new_id = self.id + 10

            # create tally clone if not present in user_tallies, update tally_ids
            new_id, tally_clone = self.create_tally_clone(new_id, ids_list,
                                                          entit_group, user_tallies_l)

            # append tally clone if created
            if tally_clone:
                tally_clones_l.append(tally_clone)
                ids_list.append(new_id)

        return ids_list, tally_clones_l

    def match_tally_simple_form(self, entit_group, model_cells_map, clone_entit_groups_d, model_cells, tally):
        """
        function for tally match and copy w/ entity group simple form definition w/ parentheses,
        i.e. (C1, ..., Cn).
        This function create/update clone_entit_groups_d, i.e. dictionary w/
        each entry corresponding to f card string of each tally clone
        """

        # parsing entit group and check match or copy with entity id
        entit_group_parsed = card_entities_parsing(entit_group[1:-1])
        clone_entit_group_parsed = {}
        current_univ = ""
        for entit_id in entit_group_parsed:
            tally_match, tally_copy, clone_entit_group_parsed = match_tally_id_entity(entit_id,
                                                                                      model_cells_map,
                                                                                      clone_entit_group_parsed)
            # check if belonging to the same universe
            if tally.type == "universe":
                if tally_match:
                    id_cell_univ = int(entit_id)
                elif tally_copy:
                    id_cell_univ = int(inverse_map_cells(model_cells_map)[entit_id])
                if tally_match or tally_copy:
                    if current_univ == "":
                        current_univ = model_cells[id_cell_univ].universe_id
                    elif current_univ != model_cells[id_cell_univ].universe_id:
                        tally_match, tally_copy = False, False
                        msg = "Tally {} has cells belonging ".format(tally.id) + \
                              "to multiple universes."
                        logger.log("error", msg)
                        break

            # exit from cycle if there is no match with any cell or parent id
            if tally_match == False and tally_copy == False:
                break

        # reconstruct string and add to clone_entit_groups_d
        for idx_copy in clone_entit_group_parsed.keys():
            entit_group_parsed = clone_entit_group_parsed[idx_copy]
            if idx_copy not in clone_entit_groups_d.keys():
                clone_entit_groups_d[idx_copy] = "(" + entit_group_parsed + ")"
            else:
                clone_entit_groups_d[idx_copy] = clone_entit_groups_d[idx_copy] + " " + "(" + entit_group_parsed + ")"

        # check number of clones created for this entity group
        if len(clone_entit_group_parsed) != len(clone_entit_groups_d):
            logger.log("info_file",
                       f"Tally {self.id} not cloned due to conflict on numbers of linked components", 10)

        return tally_match, tally_copy, clone_entit_groups_d

    def match_tally_complex_form(self, entit_group, model_cells_map, clone_entit_groups_d):
        """
         Function to check match or clone for complex entity group form definition,
         i.e. ((Si..Sn)<(Cm Cn[Ij..Ik])<(Cl Ck[Ij..Ik])...(Cp Co))
         In particular, if:
         * tally_copy is True: tally_match is False, and clone_entit_groups_d dict. is created/updated,
                                where each entry corresponds to card specification of each tally clone.
         * tally_match is True: tally_copy is False
         * tally_copy is False: it returns empty dict. for clone_entit_groups_d
        """

        # get parent map associated to child cells
        parent_map = {}
        for key, value in model_cells_map.items():
            if value != 'None':
                parent_map.setdefault(value, []).append(key)
            elif value == 'None':
                parent_map.setdefault('no_parent', []).append(key)

        # get matches & duplication of entity group by complex string parsing
        tally_match, tally_copy, clone_entit_group_parsed = analyze_complex_entit_group(entit_group, model_cells_map,
                                                                                        parent_map,
                                                                                        self.entity_type_f_card)

        # reconstruct the string and add to clone_entit_groups_d
        for idx_copy in clone_entit_group_parsed.keys():
            entit_group_parsed = clone_entit_group_parsed[idx_copy]
            if idx_copy not in clone_entit_groups_d.keys():
                clone_entit_groups_d[idx_copy] = reconstr_entit_group_parsed(entit_group_parsed, entit_group)
            else:
                clone_entit_groups_d[idx_copy] = clone_entit_groups_d[idx_copy] + " " + reconstr_entit_group_parsed(
                    entit_group_parsed, entit_group)

        # check number of clones created for this entity group
        if len(clone_entit_group_parsed) != len(clone_entit_groups_d):
            logger.log("info_file",
                       f"Tally {self.id} not cloned due to conflict on numbers of linked components", 10)

        return tally_match, tally_copy, clone_entit_groups_d

    def analyze_entity_groups(self, entit_groups_l, model_cells_map, model_cells, tally, fcheck):
        """
        analyze entity groups to check match w/ cells ids in the geom.
        it returns clone entity groups if match w/ parent ids
        """

        # default values
        clone_entit_groups_d = {}
        tally_copy = False
        tally_match = False
        # current universe to check for tally in simple form
        current_univ = ""

        # check each entity group
        for entit_group in entit_groups_l:
            # process. tally entity group in complex form or simple form w/ parhentheses
            if entit_group.startswith("(") and entit_group.endswith(")"):
                if "<" in entit_group:
                    tally_match, tally_copy, clone_entit_groups_d = self.match_tally_complex_form(entit_group,
                                                                                                  model_cells_map,
                                                                                                  clone_entit_groups_d)

                else:
                    tally_match, tally_copy, clone_entit_groups_d = self.match_tally_simple_form(entit_group,
                                                                                                 model_cells_map,
                                                                                                 clone_entit_groups_d,
                                                                                                 model_cells, tally)

            # process. tally entity group in simple form
            else:
                tally_match, tally_copy, clone_entit_groups_d = match_tally_id_entity(entit_group,
                                                                                      model_cells_map,
                                                                                      clone_entit_groups_d)
                # check if belonging to the same universe
                if tally.type == "universe" and fcheck:
                    if tally_match:
                        id_cell_univ = int(entit_group)
                    elif tally_copy:
                        id_cell_univ = int(inverse_map_cells(model_cells_map)[entit_group])
                    if tally_match or tally_copy:
                        if current_univ == "":
                            current_univ = model_cells[id_cell_univ].universe_id
                        elif current_univ != model_cells[id_cell_univ].universe_id:
                            tally_match, tally_copy = False, False
                            msg = "Tally {} has cells belonging ".format(tally.id) + \
                                  "to multiple universes."
                            logger.log("error", msg)
                            break

            # at least one group that doesn't match or cloned break for cycle
            if tally_match == False and tally_copy == False:
                break

        return tally_match, tally_copy, clone_entit_groups_d

    def update_cf_card(self, tally_match, tally_copy, tally_match_cf, tally_copy_cf,
                       clone_entit_groups_cf_d, tally_clones_l):
        """
        It processes cf card for cloned tallies.
        This approach requires duplication lead by Fn card.
        If CFn card is duplicated but Fn card is not, a warning message is printed.
        """
        # alignment between f and cf card cloned
        if tally_copy == True and tally_copy_cf == True:
            for idx_clone in range(1, len(tally_clones_l) + 1):
                cf_entry = "cf" + str(tally_clones_l[idx_clone - 1].id)
                tally_clones_l[idx_clone - 1].cf_card = cf_entry + " " + clone_entit_groups_cf_d[idx_clone]

        # discrepancy between cf card that should be duplicated and fn card not duplicated
        elif tally_copy == False and tally_copy_cf == True:
            logger.log("info_file",
                       f"Tally {self.id} not cloned due to conflict between components in Fn and Cfn card.", 10)

        # no match for cf card
        elif tally_match_cf == False and tally_copy_cf == False:
            logger.log("info_file",
                       f"Tally {self.id} not cloned, at least a component of CFn card not in geometry.", 10)

        return tally_clones_l

    def filt_mat_map(self, cell_to_mat_map, model_materials_map):
        """filter map, considering the material present in f_card_mat"""

        f_card_mat = self.f_card_mat

        # get f_card_mat if not present
        if not f_card_mat:
            # get f_card_mat (assoc. to mats) from f_card(assoc. to cells)
            f_card_mat = substit_f_card(self.f_card, convert_map_to_int(cell_to_mat_map))

        # Get material ids from f_card_ma through parsing: extract unique ids from card string.
        # Regular expression to match numbers not inside square brackets
        numbers = re.findall(r'\b(?![^\[]*\])\d+\b', f_card_mat)

        # Convert to a set to get unique numbers, then convert back to a sorted list
        m_list = sorted(set(map(int, numbers)))
        mat_ids_set = set(m_list)

        # filt mat map inv.: parent_id -> child_id
        filt_map = inverse_map_cells({key: val for key, val in model_materials_map.items() if int(key) in mat_ids_set})

        return filt_map

    def check_fm_card(self, cell_to_mat_map, material_map):
        """udpdates fm card checking match between materials and cells ids"""
        id, fm_card = self.id, self.fm_card

        # get nested list for mat ids from fm_string
        nested_l = fmstring_to_nested_list(fm_card)

        # getting reduced material maps by filtering mats in f_card_mat
        # check that ids are taken instead of rr
        reduced_material_map = self.filt_mat_map(cell_to_mat_map, material_map)

        # susbtitution if matching with mat. map
        nested_l = check_nested_l(nested_l, reduced_material_map)

        # get fm string from updated nested list
        self.fm_card = nested_list_to_fmstring(nested_l)

        return self
