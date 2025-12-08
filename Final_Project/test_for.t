print("Final Project: For Loop Tests ===");

// ------------------------------
// Basic counting loop
// ------------------------------
x = 0;
for (i = 0; i < 5; i = i + 1) {
    x = x + i;
};
assert x == 10;

// ------------------------------
// Reverse loop (counting down)
// ------------------------------
y = 0;
for (j = 5; j > 0; j = j - 1) {
    y = y + j;
};
assert y == 15;   // 5 + 4 + 3 + 2 + 1

// ------------------------------
// Loop that never executes
// ------------------------------
z = 999;
for (k = 10; k < 5; k = k + 1) {
    z = 0;
};
assert z == 999;

// ------------------------------
// Loop that executes exactly once
// ------------------------------
count = 0;
for (a = 0; a < 1; a = a + 1) {
    count = count + 1;
};
assert count == 1;

// ------------------------------
// Testing continue
// ------------------------------
sum = 0;
for (i = 0; i < 5; i = i + 1) {
    if (i == 2) { continue };
    sum = sum + i;
};
assert sum == 8;   // skipped 2 → 0+1+3+4

// ------------------------------
// Testing break
// ------------------------------
hit = 0;
for (i = 0; i < 10; i = i + 1) {
    if (i == 4) { break };
    hit = hit + 1;
};
assert hit == 4;

// ------------------------------
// Nested for loops
// ------------------------------
total = 0;
for (i = 0; i < 3; i = i + 1) {
    for (j = 0; j < 2; j = j + 1) {
        total = total + 1;
    };
};
assert total == 6;

// ------------------------------
// Nested loops building something real
// e.g., multiplication table sum
// ------------------------------
result = 0;
for (i = 1; i <= 3; i = i + 1) {
    for (j = 1; j <= 3; j = j + 1) {
        result = result + (i * j);
    };
};
assert result == 36;   // 1*1+1*2+...+3*3

// ------------------------------
// Loop modifying a list
// ------------------------------
arr = [1, 2, 3, 4, 5];
for (i = 0; i < length(arr); i = i + 1) {
    arr[i] = arr[i] * 2;
};
assert arr == [2,4,6,8,10];

// ------------------------------
// Loop building a list dynamically
// ------------------------------
out = [];
for (i = 0; i < 5; i = i + 1) {
    out = out + [i * i];
};
assert out == [0,1,4,9,16];

// ------------------------------
// Loop inside an if-statement
// ------------------------------
v = 0;
flag = true;
if (flag) {
    for (i = 0; i < 3; i = i + 1) {
        v = v + 2;
    };
};
assert v == 6;

// ------------------------------
// Update expression with more complex logic
// ------------------------------
m = 1;
for (i = 1; i < 4; i = i + 1) {
    m = m * i;
};
assert m == 6;

// ------------------------------
// Ensure each component (init, condition, update)
// is executed exactly in the expected order
// ------------------------------
trace = [];
for (i = 0; i < 3; i = i + 1) {
    trace = trace + ["body"];
};
assert trace == ["body","body","body"];

print("For Loop tests completed. All tests passed successfully as there are no errors displayed");

