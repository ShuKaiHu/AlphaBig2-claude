#big 2 class
import enumerateOptions
import gameLogic
import numpy as np
import random
import math
import itertools
import copy
from multiprocessing import Process, Pipe

def convertAvailableActions(availAcs):
    #convert from (1,0,0,1,1...) to (0, -math.inf, -math.inf, 0,0...) etc
    availAcs[np.nonzero(availAcs==0)] = -math.inf
    availAcs[np.nonzero(availAcs==1)] = 0
    return availAcs

class handPlayed:
    def __init__(self, hand, player):
        self.hand = hand
        self.player = player
        self.nCards = len(hand)
        if self.nCards <= 3:
            self.type = 1
        elif self.nCards == 5:
            if gameLogic.isStraightFlush(hand):
                self.type = 4
            elif gameLogic.isFourOfAKind(hand):
                self.type = 3
            elif gameLogic.isFullHouse(hand)[0]:
                self.type = 2
            else:
                self.type = 1

class big2Game:
    def __init__(self):
        self.reset()

    def clone(self):
        # Create a shallow instance and copy state for simulation.
        new = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                setattr(new, key, value.copy())
            elif isinstance(value, dict):
                setattr(new, key, copy.deepcopy(value))
            else:
                setattr(new, key, copy.deepcopy(value))
        return new
        
    def reset(self, hands=None, starter=None, must_play_club3=None):
        """Deal a new game.

        ADDITIVE state-injection support (M3, 2026-07). With NO arguments this
        is byte-identical to the original reset() -- same RNG consumption, same
        deal, same flags (verified by seeded-deal + full-game regression; the
        live online campaign re-imports this file and must see zero change).

        hands: FULL-DEAL injection. Either a list/tuple of 4 card-id lists
            (index 0 -> player 1) or a dict keyed {1,2,3,4} (engine players)
            or {0,1,2,3} (0-based seats, mapped seat+1). Must be an exact
            partition of card ids 1..52 into 4 hands of 13 -- raises
            ValueError otherwise. The club3Player / mustPlayClub3 logic then
            runs on the injected deal exactly as on a random one.
        starter: override which player (1-4) acts first. Default None keeps
            the club-3 holder. If you set starter to a different player while
            mustPlayClub3 stays True, the 3C holder will still be forced when
            they later gain control -- pass must_play_club3=False for
            mid-game-like openings.
        must_play_club3: override the forced-3C-opening flag (bool). Default
            None = current logic (True, forced on the 3C holder).
        """
        if hands is None:
            shuffledDeck = np.random.permutation(52) + 1
        else:
            shuffledDeck = self._validated_full_deal(hands)
        #hand out cards to each player
        self.currentHands = {}
        self.currentHands[1] = np.sort(shuffledDeck[0:13])
        self.currentHands[2] = np.sort(shuffledDeck[13:26])
        self.currentHands[3] = np.sort(shuffledDeck[26:39])
        self.currentHands[4] = np.sort(shuffledDeck[39:52])
        self.cardsPlayed = np.zeros((4,52), dtype=int)
        #who has 3C - must start and must include it in the first hand
        for i in range(52):
            if shuffledDeck[i] == 1:
                threeClubInd = i
                break
        if threeClubInd < 13:
            whoHas3C = 1
        elif threeClubInd < 26:
            whoHas3C = 2
        elif threeClubInd < 39:
            whoHas3C = 3
        else:
            whoHas3C = 4
        self.goIndex = 1
        self.handsPlayed = {}
        self.playersGo = whoHas3C
        self.passCount = 0
        self.passedThisRound = {1: False, 2: False, 3: False, 4: False}
        self.lastPlayedPlayer = whoHas3C
        self.control = 1
        self.mustPlayClub3 = True
        self.club3Player = whoHas3C
        self.actionHistory = []
        self.neuralNetworkInputs = {}
        self.neuralNetworkInputs[1] = np.zeros((412,), dtype=int)
        self.neuralNetworkInputs[2] = np.zeros((412,), dtype=int)
        self.neuralNetworkInputs[3] = np.zeros((412,), dtype=int)
        self.neuralNetworkInputs[4] = np.zeros((412,), dtype=int)
        nPlayerInd = 22*13
        nnPlayerInd = nPlayerInd + 27
        nnnPlayerInd = nnPlayerInd + 27
        #initialize number of cards
        for i in range(1,5):
            self.neuralNetworkInputs[i][nPlayerInd+12]=1
            self.neuralNetworkInputs[i][nnPlayerInd+12]=1
            self.neuralNetworkInputs[i][nnnPlayerInd+12]=1
        self.fillNeuralNetworkHand(1)
        self.fillNeuralNetworkHand(2)
        self.fillNeuralNetworkHand(3)
        self.fillNeuralNetworkHand(4)
        self.gameOver = 0
        self.rewards = np.zeros((4,))
        self.goCounter = 0
        # -- injection overrides (no-ops when reset() is called with no args) --
        if starter is not None:
            if starter not in (1, 2, 3, 4):
                raise ValueError(f"starter must be 1-4, got {starter!r}")
            self.playersGo = starter
            self.lastPlayedPlayer = starter
        if must_play_club3 is not None:
            self.mustPlayClub3 = bool(must_play_club3)

    @staticmethod
    def _normalize_hands_arg(hands):
        """Accept list/tuple of 4 hands (index 0 -> player 1) or dict keyed
        {1,2,3,4} or {0,1,2,3}; return {player(1-4): [int card ids]}."""
        if isinstance(hands, dict):
            keys = set(hands.keys())
            if keys == {1, 2, 3, 4}:
                out = {p: hands[p] for p in range(1, 5)}
            elif keys == {0, 1, 2, 3}:
                out = {s + 1: hands[s] for s in range(4)}
            else:
                raise ValueError(
                    f"hands dict must be keyed 1-4 or 0-3, got keys {sorted(keys)!r}")
        elif isinstance(hands, (list, tuple)) and len(hands) == 4:
            out = {i + 1: hands[i] for i in range(4)}
        else:
            raise ValueError("hands must be a dict of 4 hands or a list/tuple of 4 hands")
        norm = {}
        for p in range(1, 5):
            cards = [int(c) for c in out[p]]
            for c in cards:
                if not 1 <= c <= 52:
                    raise ValueError(f"player {p}: card id {c} outside 1..52")
            if len(set(cards)) != len(cards):
                raise ValueError(f"player {p}: duplicate card ids in hand {cards}")
            norm[p] = cards
        return norm

    @classmethod
    def _validated_full_deal(cls, hands):
        """Validate a FULL-DEAL injection and return it as a 52-card deck array
        laid out so reset()'s existing slicing reproduces the hands exactly."""
        norm = cls._normalize_hands_arg(hands)
        for p in range(1, 5):
            if len(norm[p]) != 13:
                raise ValueError(
                    f"full-deal injection: player {p} has {len(norm[p])} cards, need 13")
        allc = norm[1] + norm[2] + norm[3] + norm[4]
        if set(allc) != set(range(1, 53)):
            missing = sorted(set(range(1, 53)) - set(allc))
            dupes = sorted({c for c in allc if allc.count(c) > 1})
            raise ValueError(
                f"full-deal injection must partition card ids 1..52 across 4x13 "
                f"(missing={missing}, duplicated={dupes})")
        return np.array(allc, dtype=np.int64)

    @classmethod
    def restore_state(cls, hands, whose_turn, trick_cards=None, trick_owner=None,
                      passed_players=(), cards_played=None, must_play_club3=False):
        """MID-GAME state injection: reconstruct a fully playable big2Game.

        Modeled on ppo/bc_dataset._snapshot (the BC replay snapshot) but returns
        a game that can be PLAYED to completion: updateGame / step /
        returnAvailableActions / assignRewards (platform-exact scoring) all work.
        Built via __new__, so it consumes NO numpy RNG.

        hands: REMAINING cards per player -- same formats as reset(hands=...)
            (list of 4 lists, or dict keyed 1-4 / 0-3), each hand non-empty.
        whose_turn: player 1-4 to act.
        trick_cards: cards of the hand currently to beat (None = whose_turn has
            control / free lead).
        trick_owner: player 1-4 who played trick_cards (required with
            trick_cards).
        passed_players: players who have already passed this trick (same
            1-4 / 0-3 key convention as hands: ints 1-4 are engine players).
        cards_played: per-player cards ALREADY PLAYED this game -- either a
            (4,52) 0/1 mask (row s = player s+1) or the same dict/list-of-4
            format as hands. None = nothing played yet.
        must_play_club3: forced-3C flag for the restored state (default False:
            an injected mid-game state must NOT force the 3C opening).

        Validation (raises ValueError): remaining + played must exactly
        partition card ids 1..52 with no duplicates (card conservation);
        trick_cards must be among trick_owner's played cards; the actor and
        trick owner cannot be in passed_players; at most 2 passed players while
        a trick is live (3 consecutive passes would have reset the trick).

        The legacy 412-dim neuralNetworkInputs are reconstructed exactly for:
        own-hand features, opponent card counts, played big-card memory
        (both per-opponent 2s block and the global >=J record) and the current
        trick. Pass-history bits are replayed in turn order after trick_owner,
        which is exact when all live passes happened after the current trick
        play (the common case) and an approximation otherwise. The cardaware /
        v6 observation paths (obs_from_env) do not read these bits at all.
        actionHistory starts empty (history before the injection point is not
        representable). Engine DYNAMICS and SCORING are exact.
        """
        norm = cls._normalize_hands_arg(hands)
        for p in range(1, 5):
            if len(norm[p]) == 0:
                raise ValueError(f"player {p} has an empty hand -- game would be over")
        if whose_turn not in (1, 2, 3, 4):
            raise ValueError(f"whose_turn must be 1-4, got {whose_turn!r}")
        # played-cards mask
        played_mask = np.zeros((4, 52), dtype=int)
        if cards_played is not None:
            if isinstance(cards_played, np.ndarray):
                if cards_played.shape != (4, 52):
                    raise ValueError(
                        f"cards_played ndarray must be (4,52), got {cards_played.shape}")
                played_mask = (cards_played != 0).astype(int)
            else:
                pl = cls._normalize_hands_arg(cards_played)
                for p in range(1, 5):
                    for c in pl[p]:
                        played_mask[p - 1][c - 1] = 1
        # card conservation: remaining + played partition 1..52 exactly
        remaining_all = [c for p in range(1, 5) for c in norm[p]]
        played_all = [int(c) + 1 for c in np.flatnonzero(played_mask.sum(axis=0))]
        overlap = set(remaining_all) & set(played_all)
        if overlap:
            raise ValueError(f"cards both in a hand and played: {sorted(overlap)}")
        if int(played_mask.sum()) != len(played_all):
            dup = [int(c) + 1 for c in np.flatnonzero(played_mask.sum(axis=0) > 1)]
            raise ValueError(f"cards played by more than one player: {dup}")
        union = set(remaining_all) | set(played_all)
        if union != set(range(1, 53)) or len(remaining_all) + len(played_all) != 52:
            missing = sorted(set(range(1, 53)) - union)
            raise ValueError(
                f"card conservation violated: remaining({len(remaining_all)}) + "
                f"played({len(played_all)}) != 52 distinct (missing={missing})")
        # trick / pass validation
        passed = set()
        for q in passed_players:
            q = int(q)
            if q not in (1, 2, 3, 4):
                raise ValueError(f"passed player {q!r} not in 1-4")
            passed.add(q)
        if trick_cards is not None:
            if trick_owner not in (1, 2, 3, 4):
                raise ValueError("trick_owner (1-4) is required with trick_cards")
            tc = sorted(int(c) for c in trick_cards)
            for c in tc:
                if not played_mask[trick_owner - 1][c - 1]:
                    raise ValueError(
                        f"trick card {c} is not among player {trick_owner}'s played cards")
            if trick_owner == whose_turn:
                raise ValueError("trick_owner cannot be the actor while the trick is live")
            if trick_owner in passed:
                raise ValueError("trick_owner cannot have passed on its own live trick")
            if whose_turn in passed:
                raise ValueError("whose_turn has already passed -- it cannot be their go")
            if len(passed) > 2:
                raise ValueError(
                    "at most 2 players can have passed while a trick is live "
                    "(3 consecutive passes reset the trick)")
        else:
            if passed:
                raise ValueError("passed_players must be empty on a free lead (no trick)")
            tc = None

        g = cls.__new__(cls)
        g.currentHands = {p: np.sort(np.array(norm[p], dtype=np.int64))
                          for p in range(1, 5)}
        g.cardsPlayed = played_mask
        g.playersGo = whose_turn
        g.passedThisRound = {p: (p in passed) for p in range(1, 5)}
        g.passCount = len(passed)
        g.mustPlayClub3 = bool(must_play_club3)
        g.club3Player = whose_turn
        g.actionHistory = []
        g.gameOver = 0
        g.rewards = np.zeros((4,))
        g.goCounter = 0
        if tc is not None:
            g.control = 0
            g.goIndex = 2
            g.handsPlayed = {1: handPlayed(np.array(tc, dtype=np.int64), trick_owner)}
            g.lastPlayedPlayer = trick_owner
        else:
            g.control = 1
            g.goIndex = 1
            g.handsPlayed = {}
            g.lastPlayedPlayer = whose_turn
        # -- reconstruct the legacy 412-dim network inputs --
        g.neuralNetworkInputs = {p: np.zeros((412,), dtype=int) for p in range(1, 5)}
        for p in range(1, 5):
            g.fillNeuralNetworkHand(p)
        nPlayerInd = 22 * 13
        endInd = nPlayerInd + 27 + 27 + 27
        for p in range(1, 5):
            for k in (1, 2, 3):   # p's count lives in viewer (p-k)'s k-th block
                viewer = p - k if p - k >= 1 else p - k + 4
                blockInd = nPlayerInd + (k - 1) * 27
                g.neuralNetworkInputs[viewer][blockInd + g.currentHands[p].size - 1] = 1
                for c in range(45, 53):   # big cards (A/2) p has already played
                    if played_mask[p - 1][c - 1]:
                        g.neuralNetworkInputs[viewer][blockInd + 13 + (c - 45)] = 1
        for c in range(37, 53):           # global >=J played record (all viewers)
            if played_mask[:, c - 1].any():
                for p in range(1, 5):
                    g.neuralNetworkInputs[p][endInd + (c - 37)] = 1
        if tc is not None:
            g.updateNeuralNetworkInputs(np.array(tc, dtype=np.int64), trick_owner)
            if passed:                    # replay live passes in turn order
                g.passCount = 0
                q = trick_owner
                for _ in range(3):
                    q = q + 1 if q < 4 else 1
                    if q in passed:
                        g.updateNeuralNetworkPass(q)
                        g.passCount += 1
                g.passCount = len(passed)
        return g

    def fillNeuralNetworkHand(self,player):
        handOptions = gameLogic.handsAvailable(self.currentHands[player])
        sInd = 0
        self.neuralNetworkInputs[player][sInd:22*13] = 0
        for i in range(len(self.currentHands[player])):
            value = handOptions.cards[self.currentHands[player][i]].value
            suit = handOptions.cards[self.currentHands[player][i]].suit
            self.neuralNetworkInputs[player][sInd+int(value)-1] = 1
            if suit == 1:
                self.neuralNetworkInputs[player][sInd+13] = 1
            elif suit == 2:
                self.neuralNetworkInputs[player][sInd+14] = 1
            elif suit == 3:
                self.neuralNetworkInputs[player][sInd+15] = 1
            else:
                self.neuralNetworkInputs[player][sInd+16] = 1
            if handOptions.cards[self.currentHands[player][i]].inPair:
                self.neuralNetworkInputs[player][sInd+17] = 1
            if handOptions.cards[self.currentHands[player][i]].inThreeOfAKind:
                self.neuralNetworkInputs[player][sInd+18] = 1
            if handOptions.cards[self.currentHands[player][i]].inFourOfAKind:
                self.neuralNetworkInputs[player][sInd+19] = 1
            if handOptions.cards[self.currentHands[player][i]].inStraight:
                self.neuralNetworkInputs[player][sInd+20] = 1
            if handOptions.cards[self.currentHands[player][i]].inFlush:
                self.neuralNetworkInputs[player][sInd+21] = 1
            sInd += 22
    
    def updateNeuralNetworkPass(self, cPlayer):
        #this is a bit of a mess tbh, some things are unnecessary.
        phInd = 22*13 + 27 + 27 + 27 + 16
        nPlayer = cPlayer-1
        if nPlayer == 0:
            nPlayer = 4
        nnPlayer = nPlayer - 1
        if nnPlayer == 0:
            nnPlayer = 4
        nnnPlayer = nnPlayer - 1
        if nnnPlayer == 0:
            nnnPlayer = 4
        if self.passCount < 2:
            #no control - prev hands remain same
            self.neuralNetworkInputs[nPlayer][phInd+26:] = 0
            self.neuralNetworkInputs[nnPlayer][phInd+26:] = 0
            self.neuralNetworkInputs[nnnPlayer][phInd+26:] = 0
            if self.passCount == 0:
                self.neuralNetworkInputs[nPlayer][phInd+27] = 1
                self.neuralNetworkInputs[nnPlayer][phInd+27] = 1
                self.neuralNetworkInputs[nnnPlayer][phInd+27] = 1
            else:
                self.neuralNetworkInputs[nPlayer][phInd+28] = 1
                self.neuralNetworkInputs[nnPlayer][phInd+28] = 1
                self.neuralNetworkInputs[nnnPlayer][phInd+28] = 1
        else:
            #next player is gaining control.
            self.neuralNetworkInputs[nPlayer][phInd:] = 0
            self.neuralNetworkInputs[nnPlayer][phInd:] = 0
            self.neuralNetworkInputs[nnnPlayer][phInd:] = 0
            self.neuralNetworkInputs[nnnPlayer][phInd+17] = 1
    
    def updateNeuralNetworkInputs(self,prevHand, cPlayer):
        self.fillNeuralNetworkHand(cPlayer)
        nPlayer = cPlayer-1
        if nPlayer == 0:
            nPlayer = 4
        nnPlayer = nPlayer - 1
        if nnPlayer == 0:
            nnPlayer = 4
        nnnPlayer = nnPlayer - 1
        if nnnPlayer == 0:
            nnnPlayer = 4
        nCards = self.currentHands[cPlayer].size
        cardsOfNote = np.intersect1d(prevHand, np.arange(45,53))
        nPlayerInd = 22*13
        nnPlayerInd = nPlayerInd + 27
        nnnPlayerInd = nnPlayerInd + 27
        #next player
        self.neuralNetworkInputs[nPlayer][nPlayerInd:(nPlayerInd+13)] = 0
        self.neuralNetworkInputs[nPlayer][nPlayerInd+nCards-1] = 1 #number of cards
        #next next player
        self.neuralNetworkInputs[nnPlayer][nnPlayerInd:(nnPlayerInd+13)] = 0
        self.neuralNetworkInputs[nnPlayer][nnPlayerInd + nCards-1] = 1
        #next next next player
        self.neuralNetworkInputs[nnnPlayer][nnnPlayerInd:(nnnPlayerInd+13)] = 0
        self.neuralNetworkInputs[nnnPlayer][nnnPlayerInd + nCards-1] = 1
        for val in cardsOfNote:
            self.neuralNetworkInputs[nPlayer][nPlayerInd+13+(val-45)] = 1
            self.neuralNetworkInputs[nnPlayer][nnPlayerInd+13+(val-45)] = 1
            self.neuralNetworkInputs[nnnPlayer][nnnPlayerInd+13+(val-45)] = 1
        #prevHand
        phInd = nnnPlayerInd + 27 + 16
        self.neuralNetworkInputs[nPlayer][phInd:] = 0
        self.neuralNetworkInputs[nnPlayer][phInd:] = 0
        self.neuralNetworkInputs[nnnPlayer][phInd:] = 0
        self.neuralNetworkInputs[cPlayer][phInd:] = 0
        nCards = prevHand.size
        
        if nCards == 2:
            self.neuralNetworkInputs[nPlayer][nPlayerInd+21] = 1
            self.neuralNetworkInputs[nnPlayer][nnPlayerInd+21] = 1
            self.neuralNetworkInputs[nnnPlayer][nnnPlayerInd+21] = 1
            value = int(gameLogic.cardValue(prevHand[1]))
            suit = prevHand[1] % 4
            self.neuralNetworkInputs[nPlayer][phInd+19] = 1
            self.neuralNetworkInputs[nnPlayer][phInd+19] = 1
            self.neuralNetworkInputs[nnnPlayer][phInd+19] = 1
        elif nCards == 3:
            self.neuralNetworkInputs[nPlayer][nPlayerInd+22] = 1
            self.neuralNetworkInputs[nnPlayer][nnPlayerInd+22] = 1
            self.neuralNetworkInputs[nnnPlayer][nnnPlayerInd+22] = 1
            value = int(gameLogic.cardValue(prevHand[2]))
            suit = prevHand[2] % 4
            self.neuralNetworkInputs[nPlayer][phInd+20] = 1
            self.neuralNetworkInputs[nnPlayer][phInd+20] = 1
            self.neuralNetworkInputs[nnnPlayer][phInd+20] = 1
        elif nCards == 4:
            self.neuralNetworkInputs[nPlayer][nPlayerInd+23] = 1
            self.neuralNetworkInputs[nnPlayer][nnPlayerInd+23] = 1
            self.neuralNetworkInputs[nnnPlayer][nnnPlayerInd+23] = 1
            value = int(gameLogic.cardValue(prevHand[3]))
            suit = prevHand[3] % 4
            if gameLogic.isTwoPair(prevHand):
                self.neuralNetworkInputs[nPlayer][phInd+21] = 1
                self.neuralNetworkInputs[nnPlayer][phInd+21] = 1
                self.neuralNetworkInputs[nnnPlayer][phInd+21] = 1
            else:
                self.neuralNetworkInputs[nPlayer][phInd+22] = 1
                self.neuralNetworkInputs[nnPlayer][phInd+22] = 1
                self.neuralNetworkInputs[nnnPlayer][phInd+22] = 1
        elif nCards == 5:
            #import pdb; pdb.set_trace()
            if gameLogic.isStraight(prevHand):
                self.neuralNetworkInputs[nPlayer][nPlayerInd+24] = 1
                self.neuralNetworkInputs[nnPlayer][nnPlayerInd+24] = 1
                self.neuralNetworkInputs[nnnPlayer][nnnPlayerInd+24] = 1
                value = int(gameLogic.cardValue(prevHand[4]))
                suit = prevHand[4] % 4
                self.neuralNetworkInputs[nPlayer][phInd+23] = 1
                self.neuralNetworkInputs[nnPlayer][phInd+23] = 1
                self.neuralNetworkInputs[nnnPlayer][phInd+23] = 1
            if gameLogic.isFlush(prevHand):
                self.neuralNetworkInputs[nPlayer][nPlayerInd + 25] = 1
                self.neuralNetworkInputs[nnPlayer][nnPlayerInd + 25] = 1
                self.neuralNetworkInputs[nnnPlayer][nnnPlayerInd+25] = 1
                value = int(gameLogic.cardValue(prevHand[4]))
                suit = prevHand[4] % 4
                self.neuralNetworkInputs[nPlayer][phInd+24] = 1
                self.neuralNetworkInputs[nnPlayer][phInd+24] = 1
                self.neuralNetworkInputs[nnnPlayer][phInd+24] = 1
            elif gameLogic.isFullHouse(prevHand):
                self.neuralNetworkInputs[nPlayer][nPlayerInd + 26] = 1
                self.neuralNetworkInputs[nnPlayer][nnPlayerInd + 26] = 1
                self.neuralNetworkInputs[nnnPlayer][nnnPlayerInd + 26] = 1
                value = int(gameLogic.cardValue(prevHand[2]))
                suit = -1
                self.neuralNetworkInputs[nPlayer][phInd+25] = 1
                self.neuralNetworkInputs[nnPlayer][phInd+25] = 1
                self.neuralNetworkInputs[nnnPlayer][phInd+25] = 1
        else:
            value = int(gameLogic.cardValue(prevHand[0]))
            suit = prevHand[0] % 4
            self.neuralNetworkInputs[nPlayer][phInd+18] = 1
            self.neuralNetworkInputs[nnPlayer][phInd+18] = 1
            self.neuralNetworkInputs[nnnPlayer][phInd+18] = 1
        self.neuralNetworkInputs[nPlayer][phInd+value-1] = 1
        self.neuralNetworkInputs[nnPlayer][phInd+value-1] = 1
        self.neuralNetworkInputs[nnnPlayer][phInd+value-1] = 1
        if suit == 1:
            self.neuralNetworkInputs[nPlayer][phInd+13] = 1
            self.neuralNetworkInputs[nnPlayer][phInd+13] = 1
            self.neuralNetworkInputs[nnnPlayer][phInd+13] = 1
        elif suit == 2:
            self.neuralNetworkInputs[nPlayer][phInd+14] = 1
            self.neuralNetworkInputs[nnPlayer][phInd+14] = 1
            self.neuralNetworkInputs[nnnPlayer][phInd+14] = 1
        elif suit == 3:
            self.neuralNetworkInputs[nPlayer][phInd+15] = 1
            self.neuralNetworkInputs[nnPlayer][phInd+15] = 1
            self.neuralNetworkInputs[nnnPlayer][phInd+15] = 1
        elif suit == 0:
            self.neuralNetworkInputs[nPlayer][phInd+16] = 1
            self.neuralNetworkInputs[nnPlayer][phInd+16] = 1
            self.neuralNetworkInputs[nnnPlayer][phInd+16] = 1
        #general - common to all hands.
        cardsRecord = np.intersect1d(prevHand, np.arange(37,53))
        endInd = nnnPlayerInd + 27
        for val in cardsRecord:
            self.neuralNetworkInputs[1][endInd+(val-37)] = 1
            self.neuralNetworkInputs[2][endInd+(val-37)] = 1
            self.neuralNetworkInputs[3][endInd+(val-37)] = 1
            self.neuralNetworkInputs[4][endInd+(val-37)] = 1
        #no passes.
        self.neuralNetworkInputs[nPlayer][phInd+26] = 1
        self.neuralNetworkInputs[nnPlayer][phInd+26] = 1
        self.neuralNetworkInputs[nnnPlayer][phInd+26] = 1
        self.neuralNetworkInputs[nPlayer][phInd+27:] = 0
        self.neuralNetworkInputs[nnPlayer][phInd+27:] = 0
        self.neuralNetworkInputs[nnnPlayer][phInd+27:] = 0                
    
    def updateGame(self, option, nCards=0):
        self.goCounter += 1
        if option == -1:
            #they pass
            cPlayer = self.playersGo
            passed_snapshot = np.array(
                [1 if self.passedThisRound[i] else 0 for i in range(1, 5)],
                dtype=np.float32,
            )
            self.actionHistory.append(
                {
                    "player": cPlayer,
                    "hand": None,
                    "pass": True,
                    "forced_skip": False,
                    "control_break": bool(self.control == 1),
                    "passed_snapshot": passed_snapshot,
                }
            )
            self.updateNeuralNetworkPass(cPlayer)
            if not self.passedThisRound[cPlayer]:
                self.passedThisRound[cPlayer] = True
                self.passCount += 1
            if self.passCount == 3:
                self.control = 1
                self.passCount = 0
                self.passedThisRound = {1: False, 2: False, 3: False, 4: False}
                self.playersGo = self.lastPlayedPlayer
                return
            self.playersGo += 1
            if self.playersGo == 5:
                self.playersGo = 1
            # skip players who already passed this round
            while self.control == 0 and self.passedThisRound[self.playersGo]:
                skip_player = self.playersGo
                passed_snapshot = np.array(
                    [1 if self.passedThisRound[i] else 0 for i in range(1, 5)],
                    dtype=np.float32,
                )
                self.actionHistory.append(
                    {
                        "player": skip_player,
                        "hand": None,
                        "pass": True,
                        "forced_skip": True,
                        "control_break": False,
                        "passed_snapshot": passed_snapshot,
                    }
                )
                self.playersGo += 1
                if self.playersGo == 5:
                    self.playersGo = 1
            return
        cPlayer = self.playersGo
        passed_snapshot = np.array(
            [1 if self.passedThisRound[i] else 0 for i in range(1, 5)],
            dtype=np.float32,
        )
        self.passedThisRound[cPlayer] = False
        if isinstance(option, (list, tuple, np.ndarray)):
            handToPlay = np.array(option, dtype=int)
        else:
            if nCards == 1:
                handToPlay = np.array([self.currentHands[self.playersGo][option]])
            elif nCards == 2:
                handToPlay = self.currentHands[self.playersGo][enumerateOptions.inverseTwoCardIndices[option]]
            elif nCards == 4:
                handToPlay = self.currentHands[self.playersGo][enumerateOptions.inverseFourCardIndices[option]]
            else:
                handToPlay = self.currentHands[self.playersGo][enumerateOptions.inverseFiveCardIndices[option]]
        for i in handToPlay:
            self.cardsPlayed[self.playersGo-1][i-1] = 1
        self.actionHistory.append(
            {
                "player": cPlayer,
                "hand": handToPlay.copy(),
                "pass": False,
                "forced_skip": False,
                "control_break": bool(self.control == 1),
                "passed_snapshot": passed_snapshot,
            }
        )
        self.handsPlayed[self.goIndex] = handPlayed(handToPlay, self.playersGo)
        self.control = 0
        self.goIndex += 1
        self.lastPlayedPlayer = cPlayer
        if self.mustPlayClub3 and cPlayer == self.club3Player:
            if 1 in handToPlay:
                self.mustPlayClub3 = False
        self.currentHands[self.playersGo] = np.setdiff1d(self.currentHands[self.playersGo],handToPlay)
        if self.currentHands[self.playersGo].size == 0:
            self.assignRewards()
            self.gameOver = 1
            return
        self.updateNeuralNetworkInputs(handToPlay, self.playersGo)
        self.playersGo += 1
        if self.playersGo == 5:
            self.playersGo = 1
        # skip players who already passed this round
        while self.control == 0 and self.passedThisRound[self.playersGo]:
            skip_player = self.playersGo
            passed_snapshot = np.array(
                [1 if self.passedThisRound[i] else 0 for i in range(1, 5)],
                dtype=np.float32,
            )
            self.actionHistory.append(
                {
                    "player": skip_player,
                    "hand": None,
                    "pass": True,
                    "forced_skip": True,
                    "control_break": False,
                    "passed_snapshot": passed_snapshot,
                }
            )
            self.playersGo += 1
            if self.playersGo == 5:
                self.playersGo = 1
            
    def assignRewards(self):
        totCardsLeft = 0
        winner = None
        for i in range(1,5):
            nC = self.currentHands[i].size
            if nC == 0:
                winner = i
            else:
                self.rewards[i-1] = -1 * nC
                totCardsLeft += nC
        self.rewards[winner-1] = totCardsLeft

        # Apply Big2 doubling rules.
        loser_multipliers = {}
        for i in range(1,5):
            if i == winner:
                continue
            loser_multipliers[i] = self._hand_multiplier(self.currentHands[i])

        for i in range(1,5):
            if i == winner:
                continue
            self.rewards[i-1] *= loser_multipliers[i]
        # Winner earns the sum of losers' losses.
        loser_sum = 0.0
        for i in range(1, 5):
            if i == winner:
                continue
            loser_sum += self.rewards[i - 1]
        self.rewards[winner-1] = -1 * loser_sum

        winner_last_hand = self.handsPlayed[self.goIndex - 1].hand
        winner_multiplier = self._winner_finish_multiplier(winner_last_hand)
        if winner_multiplier > 1:
            self.rewards *= winner_multiplier

    def _count_twos(self, hand):
        if hand.size == 0:
            return 0
        values = np.ceil(hand / 4).astype(int)
        return int(np.sum(values == 13))

    def _has_four_of_a_kind(self, hand):
        if hand.size < 4:
            return False
        hand_options = gameLogic.handsAvailable(hand, nC=4)
        return len(hand_options.fourOfAKinds) > 0

    def _has_straight_flush(self, hand):
        if hand.size < 5:
            return False
        for combo in itertools.combinations(hand, 5):
            if gameLogic.isStraightFlush(np.array(combo)):
                return True
        return False

    def _winner_finish_multiplier(self, hand):
        """Doubling from the WINNER's finishing combo (distinct from the losers'
        held-hand multiplier). Reverse-engineered to 100% match (4412/4412) against
        神來也's showScore settlement:
          - x2 if the finishing combo is a bomb (four-of-a-kind or straight-flush)
          - x2 if it contains a 2 acting as a PRINCIPAL card; stacks with the bomb x2
          - a 2 that is only INCIDENTAL does NOT count: the wheel A-2-3-4-5, the
            pair in a full house (KKK22), or the kicker of a four-of-a-kind (AAAA2).
        The old code reused _hand_multiplier here, which double-counts (pair of 2s
        -> x4 not x2) and counts incidental 2s -- wrong on 1.3% of games (always
        over-doubling the value/exhaustive-solve label for a 2/bomb finish)."""
        hand = np.asarray(hand)
        mult = 1
        if self._has_four_of_a_kind(hand) or self._has_straight_flush(hand):
            mult *= 2
        ranks = ((hand - 1) // 4).astype(int)          # 12 == '2'
        if 12 in ranks and not self._two_is_incidental(hand, ranks):
            mult *= 2
        return mult

    def _two_is_incidental(self, hand, ranks):
        rset = set(int(r) for r in ranks)
        if rset == {0, 1, 2, 11, 12}:                  # wheel A-2-3-4-5: 2 plays low
            return True
        if len(hand) == 5:
            counts = {r: int(np.sum(ranks == r)) for r in rset}
            two_n = counts.get(12, 0)
            # four-of-a-kind + lone 2 kicker, or full house with 2 as the pair
            if 4 in counts.values() and two_n == 1:
                return True
            trip = [r for r, c in counts.items() if c == 3]
            if trip and trip[0] != 12 and two_n == 2:
                return True
        return False

    def _hand_multiplier(self, hand, count_hand_size=True):
        multiplier = 1
        num_twos = self._count_twos(hand)
        if num_twos > 0:
            multiplier *= 2 ** num_twos
        if self._has_four_of_a_kind(hand):
            multiplier *= 2
        if self._has_straight_flush(hand):
            multiplier *= 2
        if count_hand_size and hand.size >= 10:
            multiplier *= 2
        return multiplier
        
    def randomOption(self):
        available_actions = self.returnAvailableActions()
        valid_actions = np.flatnonzero(available_actions == 1)
        if valid_actions.size == 0:
            return -1
        action = int(np.random.choice(valid_actions))
        if action == enumerateOptions.passInd:
            return -1
        opt, nCards = enumerateOptions.getOptionNC(action)
        return (opt, nCards)

    def _restrict_singles_for_one_card_rule(self, availableActions):
        """House rule (matches the online platform / vision-layer's now-fixed
        _apply_one_card_rule in alpha_big2_wrapper.py -- this brings self-play/
        rollout in line with it, 2026-07-08): if the EFFECTIVE NEXT ACTOR (the
        first still-active downstream seat in turn order) has exactly 1 card
        remaining, restrict my singles to only the HIGHEST legal single -- never
        hand them the trick by playing low. Applies whether I'm leading fresh or
        following: a low lead can be beaten and taken by a 1-card opponent's
        higher single just as easily as a low follow can.

        A still-active seat with 2+ cards is a genuine buffer -- they get the
        real next decision and may absorb/beat the trick themselves, so a
        1-card threat further downstream isn't yet live. Stop at the first
        still-active seat; don't scan past it (the exact bug this mirrors the
        fix for: scanning all downstream seats over-restricted play whenever a
        LATER seat had 1 card behind an active, not-yet-passed buffer).

        Before this engine-side fix, self-play/rollout/BC-corpus-replay never
        saw this restriction at all (this method didn't exist) -- only the
        online vision-layer bridge enforced it, so training never learned the
        behavior and had to have it force-masked only at deployment time.
        """
        single_idx = np.flatnonzero(availableActions[:52] == 1)
        if len(single_idx) <= 1:
            return availableActions
        following = (self.control == 0)
        p = self.playersGo
        for _ in range(3):
            p = p + 1 if p < 4 else 1
            if self.currentHands[p].size == 0:
                continue
            if following and self.passedThisRound[p]:
                continue
            if self.currentHands[p].size == 1:
                highest = int(single_idx.max())
                for idx in single_idx:
                    if int(idx) != highest:
                        availableActions[int(idx)] = 0
            break
        return availableActions

    def _action_includes_card(self, action, card_id):
        opt, nCards = enumerateOptions.getOptionNC(action)
        if nCards == 0:
            return False
        if isinstance(opt, (list, tuple, np.ndarray)):
            return card_id in opt
        if nCards == 1:
            return self.currentHands[self.playersGo][opt] == card_id
        if nCards == 2:
            return card_id in self.currentHands[self.playersGo][enumerateOptions.inverseTwoCardIndices[opt]]
        if nCards == 5:
            return card_id in self.currentHands[self.playersGo][enumerateOptions.inverseFiveCardIndices[opt]]
        return False
            
    def returnAvailableActions(self):
    
        currHand = self.currentHands[self.playersGo]
        availableActions = np.zeros((enumerateOptions.passInd + 1,))
        if currHand.size == 0:
            return availableActions
        
        if self.control == 0:
            prevHand = self.handsPlayed[self.goIndex-1].hand
            nCardsToBeat = len(prevHand)
            if self.passedThisRound[self.playersGo]:
                return availableActions

            handOptions = gameLogic.handsAvailable(currHand)

            if nCardsToBeat == 1:
                options = enumerateOptions.oneCardOptions(currHand, prevHand,1)
            elif nCardsToBeat == 2:
                options = enumerateOptions.twoCardOptions(handOptions, prevHand, 1)
            elif nCardsToBeat == 5:
                if gameLogic.isStraightFlush(prevHand):
                    options = enumerateOptions.fiveCardOptions(handOptions, prevHand, 4)
                elif gameLogic.isFourOfAKind(prevHand):
                    options = enumerateOptions.fiveCardOptions(handOptions, prevHand, 3)
                elif gameLogic.isFullHouse(prevHand)[0]:
                    options = enumerateOptions.fiveCardOptions(handOptions, prevHand, 2)
                else:
                    options = enumerateOptions.fiveCardOptions(handOptions, prevHand, 1)
            else:
                options = -1

            if isinstance(options, int): #no options - must pass
                options = np.array([], dtype=int)

            for option in options:
                if nCardsToBeat == 1:
                    cards = [currHand[option]]
                elif nCardsToBeat == 2:
                    card_inds = enumerateOptions.inverseTwoCardIndices[option]
                    cards = currHand[card_inds]
                else:
                    card_inds = enumerateOptions.inverseFiveCardIndices[option]
                    cards = currHand[card_inds]
                index = enumerateOptions.action_index_from_cards(cards)
                availableActions[index] = 1

            # Straight flush and four-of-a-kind can beat any hand type,
            # including a single card (e.g. spade 2).
            prev_is_straight_flush = (nCardsToBeat == 5 and gameLogic.isStraightFlush(prevHand))
            prev_is_four_kind = (nCardsToBeat == 5 and gameLogic.isFourOfAKind(prevHand))

            if not prev_is_straight_flush:
                four_kind_options = enumerateOptions.fourOfAKindOnlyOptions(
                    handOptions, prevHand if prev_is_four_kind else None
                )
                if not isinstance(four_kind_options, int):
                    for option in four_kind_options:
                        card_inds = enumerateOptions.inverseFiveCardIndices[option]
                        cards = currHand[card_inds]
                        index = enumerateOptions.action_index_from_cards(cards)
                        availableActions[index] = 1

            straight_flush_options = enumerateOptions.straightFlushOnlyOptions(
                handOptions, prevHand if prev_is_straight_flush else None
            )
            if not isinstance(straight_flush_options, int):
                for option in straight_flush_options:
                    card_inds = enumerateOptions.inverseFiveCardIndices[option]
                    cards = currHand[card_inds]
                    index = enumerateOptions.action_index_from_cards(cards)
                    availableActions[index] = 1

            # Voluntary pass is ALWAYS legal when following (control == 0) and the
            # player has not already passed this round. Holding cards/combos by
            # passing on a beatable hand is core Big 2 strategy — without this the
            # agent is FORCED to beat whenever it can, so MCTS never even explores
            # PASS and breaks pairs/straights/full-houses to dump a forced play.
            # (The early returns above — empty hand, already passed — correctly
            # leave the mask all-zeros, i.e. a forced pass with no choice.)
            availableActions[enumerateOptions.passInd] = 1
            return self._restrict_singles_for_one_card_rule(availableActions)
        
        
        else: #player has control.
            handOptions = gameLogic.handsAvailable(currHand)
            oneCardOptions = enumerateOptions.oneCardOptions(currHand)
            twoCardOptions = enumerateOptions.twoCardOptions(handOptions)
            fiveCardOptions = enumerateOptions.fiveCardOptions(handOptions)
            
            if not isinstance(oneCardOptions, int):
                for option in oneCardOptions:
                    cards = [currHand[option]]
                    index = enumerateOptions.action_index_from_cards(cards)
                    availableActions[index] = 1
                
            if not isinstance(twoCardOptions, int):
                for option in twoCardOptions:
                    card_inds = enumerateOptions.inverseTwoCardIndices[option]
                    cards = currHand[card_inds]
                    index = enumerateOptions.action_index_from_cards(cards)
                    availableActions[index] = 1
                    
            if not isinstance(fiveCardOptions, int):
                for option in fiveCardOptions:
                    card_inds = enumerateOptions.inverseFiveCardIndices[option]
                    cards = currHand[card_inds]
                    index = enumerateOptions.action_index_from_cards(cards)
                    availableActions[index] = 1
                    
            if self.mustPlayClub3 and self.playersGo == self.club3Player:
                filtered = np.zeros_like(availableActions)
                for action in np.flatnonzero(availableActions == 1):
                    if action == enumerateOptions.passInd:
                        continue
                    if self._action_includes_card(int(action), 1):
                        filtered[action] = 1
                availableActions = filtered
            return self._restrict_singles_for_one_card_rule(availableActions)

    def step(self, action):
        opt, nC = enumerateOptions.getOptionNC(action)
        self.updateGame(opt, nC)
        if self.gameOver == 0:
            reward = 0
            done = False
            info = None
        else:
            reward = self.rewards
            done = True
            info = {}
            info['numTurns'] = self.goCounter
            info['rewards'] = self.rewards
            #what else is worth monitoring?            
            self.reset()
        return reward, done, info
    
    def getCurrentState(self):
        return self.playersGo, self.neuralNetworkInputs[self.playersGo].reshape(1,412), convertAvailableActions(self.returnAvailableActions()).reshape(1,enumerateOptions.passInd + 1)
        
        
        
#now create a vectorized environment
def worker(remote, parent_remote):
    parent_remote.close()
    game = big2Game()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            reward, done, info = game.step(data)
            remote.send((reward, done, info))
        elif cmd == 'reset':
            game.reset()
            pGo, cState, availAcs = game.getCurrentState()
            remote.send((pGo,cState))
        elif cmd == 'getCurrState':
            pGo, cState, availAcs = game.getCurrentState()
            remote.send((pGo, cState, availAcs))
        elif cmd == 'close':
            remote.close()
            break
        else:
            print("Invalid command sent by remote")
            break
        

class vectorizedBig2Games(object):
    def __init__(self, nGames):
        
        self.waiting = False
        self.closed = False
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nGames)])
        self.ps = [Process(target=worker, args=(work_remote, remote)) for (work_remote, remote) in zip(self.work_remotes, self.remotes)]
        
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()
            
    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True
        
    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        rewards, dones, infos = zip(*results)
        return rewards, dones, infos
    
    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
        
    def currStates_async(self):
        for remote in self.remotes:
            remote.send(('getCurrState', None))
        self.waiting = True
        
    def currStates_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        pGos, currStates, currAvailAcs = zip(*results)
        return np.stack(pGos), np.stack(currStates), np.stack(currAvailAcs)
    
    def getCurrStates(self):
        self.currStates_async()
        return self.currStates_wait()
    
    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
