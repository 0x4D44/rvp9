#[path = "/Users/md/worktrees/rvp9/20261017-TSK-HUM-requalify-post-fix-candidate/crates/rvp9-harness/src/bdrate.rs"]
mod bdrate;

fn main() {
    let anchor = [
        (1152.068, 25.670945),
        (1174.084, 25.723668),
        (1207.220, 25.767877),
        (1253.696, 25.822174),
        (1326.328, 25.884390),
        (1569.352, 26.307338),
    ];
    let test = [
        (1151.356, 25.697093),
        (1175.428, 25.747508),
        (1207.796, 25.794290),
        (1251.228, 25.841255),
        (1326.504, 25.908804),
        (1568.004, 26.298732),
    ];
    println!("{:.12}", bdrate::bdrate(&anchor, &test));
}
