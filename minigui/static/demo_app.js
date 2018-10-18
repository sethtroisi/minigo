define(["require", "exports", "./app", "./base", "./board", "./heat_map", "./layer", "./log", "./winrate_graph", "./util"], function (require, exports, app_1, base_1, board_1, heat_map_1, lyr, log_1, winrate_graph_1, util_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    const HUMAN = 'Human';
    const MINIGO = 'Minigo';
    class DemoApp extends app_1.App {
        constructor() {
            super();
            this.playerElems = [];
            this.winrateGraph = new winrate_graph_1.WinrateGraph('winrate-graph');
            this.log = new log_1.Log('log', 'console');
            this.connect().then(() => {
                this.mainBoard = new board_1.ClickableBoard('main-board', this.size, [lyr.Label, lyr.BoardStones, [lyr.Variation, 'pv'], lyr.Annotations]);
                let boards = [this.mainBoard];
                let searchElem = util_1.getElement('search-board');
                if (searchElem) {
                    boards.push(new board_1.Board(searchElem, this.size, [[lyr.Caption, 'search'], lyr.BoardStones, [lyr.Variation, 'search']]));
                }
                let nElem = util_1.getElement('n-board');
                if (nElem) {
                    boards.push(new board_1.Board(nElem, this.size, [[lyr.Caption, 'N'], [lyr.HeatMap, 'n', heat_map_1.heatMapN], lyr.BoardStones]));
                }
                let dqElem = util_1.getElement('dq-board');
                if (dqElem) {
                    boards.push(new board_1.Board('dq-board', this.size, [[lyr.Caption, 'ΔQ'], [lyr.HeatMap, 'dq', heat_map_1.heatMapDq], lyr.BoardStones]));
                }
                this.init(boards);
                this.mainBoard.onClick((p) => {
                    this.playMove(this.toPlay, p);
                });
                this.initButtons();
                this.winrateGraph.onMoveChanged((moveNum) => {
                    let position;
                    if (moveNum == null) {
                        position = this.positionHistory[this.positionHistory.length - 1];
                    }
                    else {
                        position = this.positionHistory[moveNum];
                    }
                    if (position == this.activePosition) {
                        return;
                    }
                    this.activePosition = position;
                    this.updateBoards(position);
                });
                this.log.onConsoleCmd((cmd) => {
                    this.gtp.send(cmd).then(() => { this.log.scroll(); });
                });
                this.gtp.onText((line) => { this.log.log(line, 'log-cmd'); });
                this.newGame();
            });
        }
        initButtons() {
            util_1.getElement('pass').addEventListener('click', () => {
                if (this.mainBoard.enabled) {
                    this.playMove(this.toPlay, 'pass');
                }
            });
            util_1.getElement('reset').addEventListener('click', () => {
                this.gtp.newSession();
                this.newGame();
            });
            let initPlayerButton = (color, elemId) => {
                let elem = util_1.getElement(elemId);
                this.playerElems[color] = elem;
                elem.addEventListener('click', () => {
                    if (elem.innerText == HUMAN) {
                        elem.innerText = MINIGO;
                    }
                    else {
                        elem.innerText = HUMAN;
                    }
                    this.onPlayerChanged();
                });
            };
            initPlayerButton(base_1.Color.Black, 'black-player');
            initPlayerButton(base_1.Color.White, 'white-player');
        }
        newGame() {
            super.newGame();
            this.log.clear();
            this.winrateGraph.clear();
        }
        onPlayerChanged() {
            if (this.engineBusy || this.gameOver) {
                return;
            }
            if (this.playerElems[this.toPlay].innerText == MINIGO) {
                this.mainBoard.enabled = false;
                this.engineBusy = true;
                this.gtp.send('genmove').then((move) => {
                    this.engineBusy = false;
                    this.gtp.send('gamestate');
                });
            }
            else {
                this.mainBoard.enabled = true;
            }
        }
        onGameState(msg) {
            super.onGameState(msg);
            this.log.scroll();
            this.winrateGraph.setWinrate(msg.moveNum, msg.q);
            this.onPlayerChanged();
        }
        playMove(color, move) {
            let colorStr = color == base_1.Color.Black ? 'b' : 'w';
            let moveStr;
            if (move == 'pass') {
                moveStr = move;
            }
            else if (move == 'resign') {
                throw new Error('resign not yet supported');
            }
            else {
                let row = this.size - move.row;
                let col = base_1.COL_LABELS[move.col];
                moveStr = `${col}${row}`;
            }
            this.gtp.send(`play ${colorStr} ${moveStr}`).then(() => {
                this.gtp.send('gamestate');
            });
        }
        onGameOver() {
            this.gtp.send('final_score').then((result) => {
                this.log.log(util_1.toPrettyResult(result));
                this.log.scroll();
            });
        }
    }
    new DemoApp();
});
//# sourceMappingURL=demo_app.js.map