define(["require", "exports", "./util"], function (require, exports, util_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    class Log {
        constructor(logElemId, consoleElemId = null) {
            this.cmdHandler = null;
            this.logElem = util_1.getElement(logElemId);
            if (consoleElemId) {
                this.consoleElem = util_1.getElement(consoleElemId);
                this.consoleElem.addEventListener('keypress', (e) => {
                    if (e.keyCode == 13) {
                        let cmd = this.consoleElem.innerText.trim();
                        if (cmd != '' && this.cmdHandler) {
                            this.cmdHandler(cmd);
                        }
                        this.consoleElem.innerHTML = '';
                        e.preventDefault();
                        return false;
                    }
                });
            }
        }
        log(msg, className = '') {
            let child;
            if (typeof msg == 'string') {
                if (msg == '') {
                    msg = ' ';
                }
                child = document.createElement('div');
                child.innerText = msg;
                if (className != '') {
                    child.className = className;
                }
            }
            else {
                child = msg;
            }
            this.logElem.appendChild(child);
        }
        clear() {
            this.logElem.innerHTML = '';
        }
        scroll() {
            if (this.logElem.lastElementChild) {
                this.logElem.lastElementChild.scrollIntoView();
            }
        }
        onConsoleCmd(cmdHandler) {
            this.cmdHandler = cmdHandler;
        }
    }
    exports.Log = Log;
});
//# sourceMappingURL=log.js.map