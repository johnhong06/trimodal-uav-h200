"""평가 고리의 열린 파일 수 시험 — 2026-11-11 H200 작업 A 1 차 실패 ('received 0 items of ancdata') 의 회귀 시험.

DataLoader 작업자가 보낸 텐서는 공유 메모리라 살아 있는 동안 파일 핸들을 하나씩 쥔다. 평가가 정답 텐서를 복사하지 않고 쥐고 있으면
test 1,950 장에서 핸들이 3,900 개 쌓여 열린 파일 한도 (컨테이너 1,024) 를 넘는다. smoke 는 64 장이라 이 문제를 보지 못했다.
여기서는 한도를 1,024 로 낮추고, 모형과 데이터만 가짜로 바꿔 (저자 Dataset 과 같은 형식) src/evaluate.py 의 main() 을 그대로 부른다.
모형 · 데이터 · GPU 가 필요 없고 1 분 안에 끝난다.

  python tests/test_eval_fd.py <작업 폴더>
"""
import os
import resource
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import torch  # noqa: E402

import data as D  # noqa: E402
import evaluate as E  # noqa: E402
import model as M  # noqa: E402

N, LIM = 1950, 1024


class Fake(torch.utils.data.Dataset):
    def __init__(self, n):
        self.stems = [("frame_" if i % 3 else "nframe_") + f"{i:05d}" for i in range(n)]

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        return torch.zeros(5, 8, 8), {"boxes": torch.tensor([[1.0, 1.0, 20.0, 20.0], [30.0, 30.0, 60.0, 50.0]]),
                                      "labels": torch.tensor([1, 1], dtype=torch.int64)}


class Net(torch.nn.Module):
    def forward(self, imgs):
        return [{"boxes": torch.tensor([[1.0, 1.0, 20.0, 20.0]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([1])}
                for _ in imgs]


D.make = lambda split, limit=0: Fake(N)
M.build = lambda arm, *a, **k: Net()
M.count_params = lambda model: {}
work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
os.makedirs(work, exist_ok=True)
ck = os.path.join(work, "fake.pt")
torch.save({}, ck)
hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
resource.setrlimit(resource.RLIMIT_NOFILE, (min(LIM, hard), hard))
sys.argv = ["evaluate.py", "--arm", "paper", "--ckpt", ck, "--split", "test", "--out", os.path.join(work, "fd", "eval_test"), "--device", "cpu"]
try:
    rc = E.main()
    msg = None if rc == 0 else f"종료 코드 {rc}"
except Exception as ex:                                            # noqa: BLE001
    msg = f"{type(ex).__name__}: {ex}"
print(("PASS " if msg is None else "FAIL ") + f"평가 {N:,} 장 · 열린 파일 한도 {min(LIM, hard)}" + ("" if msg is None else f" — {msg}"), flush=True)
sys.exit(0 if msg is None else 1)
