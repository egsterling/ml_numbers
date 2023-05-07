// const { askForExtensionDescription } = require("generator-code/generators/app/prompts");

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const CANVAS_WIDTH = canvas.width;
const CANVAS_HEIGHT = canvas.height;
const BLOCK_WIDTH = 20;
const BLOCK_HEIGHT = 20;
const WIDTH_IN_BLOCKS = CANVAS_WIDTH/BLOCK_WIDTH;
const HEIGHT_IN_BLOCKS = CANVAS_HEIGHT/BLOCK_HEIGHT;
const numDict = {
    10: "A",
    11: "B",
    12: "C",
    13: "D",
    14: "E",
    15: "F"
}
const sender = document.getElementById("sender");
const output = document.getElementById("output");


let posX;
let posY;

let grid = Array.from({length: WIDTH_IN_BLOCKS}, () => Array.from({length: HEIGHT_IN_BLOCKS}, () => (0)));

let mouseDown = false;

// [min, max)
function randomInt(min, max) {
    return Math.floor(Math.random() * (max - min)) + min;
}

// function addToFifteen(a, b) {
//     let sum = a + b;
//     a = (sum <= 15 ? sum : 15);
//     return a;
// }

function addToHundred(a, b) {
    let sum = a + b;
    a = (sum <= 100 ? sum : 100);
    return a;
}

// function convertNum(a) {
//     return a <= 9 ? a : numDict[a];
// }

//  && !grid[xInd][yInd] && 

function drawIf(ctx, posX, posY, grid, parentX, parentY) {
    xInd = posX / BLOCK_WIDTH;
    yInd = posY / BLOCK_HEIGHT;
    xPInd = parentX / BLOCK_WIDTH;
    yPInd = parentY / BLOCK_HEIGHT;
    if(!(xInd >= WIDTH_IN_BLOCKS) && !(xInd < 0) && !(yInd >= HEIGHT_IN_BLOCKS) && !(yInd < 0) && grid[xPInd][yPInd] != 100) {
        let r = randomInt(35, 65);
        grid[xInd][yInd] = addToHundred(grid[xInd][yInd], r);
        let a = grid[xInd][yInd];
        // ctx.fillStyle = "#" + a + a + a;
        ctx.fillStyle = "hsl(0, 0%, " + a + "%)"
        ctx.fillRect(posX, posY, BLOCK_WIDTH, BLOCK_HEIGHT);
    }
}

function drawOnce(ctx, posX, posY, grid) {
    ctx.fillStyle = "white";
    ctx.fillRect(posX, posY, BLOCK_WIDTH, BLOCK_HEIGHT);
    drawIf(ctx, posX, posY - BLOCK_HEIGHT, grid, posX, posY);
    drawIf(ctx, posX + BLOCK_WIDTH, posY, grid, posX, posY);
    drawIf(ctx, posX, posY + BLOCK_HEIGHT, grid, posX, posY);
    drawIf(ctx, posX - BLOCK_WIDTH, posY, grid, posX, posY);
    grid[posX / BLOCK_WIDTH][posY / BLOCK_HEIGHT] = 100;
}

function process() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    canvas.addEventListener("mousedown", (event) => {
        mouseDown = true;
        posX = event.clientX - (event.clientX % BLOCK_WIDTH);
        posY = event.clientY - (event.clientY % BLOCK_HEIGHT);
        drawOnce(ctx, posX, posY, grid);
    });
    
    canvas.addEventListener("mousemove", (event) => {
        // console.log("move");
        if(mouseDown) {
            console.log("move");
            posX = event.clientX - (event.clientX % BLOCK_WIDTH);
            posY = event.clientY - (event.clientY % BLOCK_HEIGHT);
            drawOnce(ctx, posX, posY, grid);
        }
    });
    
    document.addEventListener("mouseup", () => {
        mouseDown = false;
    });

    sender.addEventListener("click", () => {
        let json = JSON.stringify(grid);

        axios.post("api/value/", json, {
            headers: {
                'Content-Type': 'application/json'
            }})
            .then((response) => {
                console.log("Hello");
                const most_likely = Object.keys(response.data).reduce((a, b) => response.data[a] > response.data[b] ? a : b);
                for (const [key, value] of Object.entries(response.data)) {
                    output.innerHTML += `<div>${key}: ${value}</div>`;
                }
                output.innerHTML += "<p>Most likely: " + most_likely + "<p>";

                
            })
            .catch((error) => {
                console.error(error);
            });
    });
}

window.addEventListener("load", process);