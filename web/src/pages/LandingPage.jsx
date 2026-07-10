import { Link } from "react-router-dom";

import { useAuth } from "../auth";
import BrandLogo from "../components/BrandLogo";
import { Badge, Button } from "../components/ui";

const pipelineSteps = [
    ["Đầu vào", "Video ngắn"],
    ["Phát hiện phương tiện", "YOLOv5m phát hiện phương tiện"],
    ["Theo vết", "BoT-SORT/ReID theo vết phương tiện qua nhiều khung hình"],
    ["Phát hiện biển số", "YOLOv8 OBB phát hiện biển số"],
    ["Bộ phân loại chất lượng", "Quyết định 1 biển số có đủ tốt để được OCR"],
    ["Nhận dạng OCR", "Small-LPR-Line-CTC nhận dạng biển số"],
    [
        "Khối tổng hợp đa khung hình",
        "Biểu quyết trên các biển số của cùng 1 track, từ đó đưa ra kết quả cuối cùng",
    ],
    ["Kết quả kiểm tra", "Ảnh phương tiện, ảnh biển số, kết quả OCR"],
];

const metrics = [
    ["YOLOv8-OBB mAP50", "98.29%", "đánh giá bộ phát hiện biển số"],
    ["SmallLPR-Line-CTC val acc", "95,01%​", "tập validation OCR nội bộ"],
    ["Quality router acc", "84,71​%", "lọc biển số phù hợp OCR"],
];

const surfaces = [
    [
        "Xử lý video",
        "Tải video, theo dõi tiến trình SSE, kiểm tra kết quả biển số và độ tin cậy.",
    ],
    [
        "Xử lí trích đoạn",
        "Kết nối video streaming hoặc tải lên video dài, đánh dấu cửa sổ ngắn để phân tích nhanh.",
    ],
    [
        "Kho lịch sử kết quả",
        "Truy xuất phiên xử lý, bản ghi nhận dạng, ảnh phương tiện và ảnh biển số đã lưu.",
    ],
];

function getPrimaryCta(user, loading) {
    if (loading) return { to: "/login", label: "Đang kiểm tra phiên" };
    if (user) return { to: "/dashboard", label: "Vào hệ thống ALPR" };
    return { to: "/register", label: "Tạo tài khoản" };
}

export default function LandingPage() {
    const { user, loading } = useAuth();
    const primaryCta = getPrimaryCta(user, loading);

    return (
        <main className="app-shell min-h-screen">
            <header className="app-topbar">
                <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-3 sm:px-6">
                    <a
                        href="/"
                        className="brand-home-link flex min-w-0 items-center gap-3"
                        aria-label="Về trang chủ"
                    >
                        <span className="brand-mark">
                            <BrandLogo />
                        </span>
                        <span className="truncate text-sm font-bold sm:text-base">
                            ALPR Việt Nam
                        </span>
                    </a>
                    <nav className="ml-auto hidden items-center gap-5 text-sm font-semibold text-[var(--color-text-muted)] md:flex">
                        <a
                            href="#pipeline"
                            className="hover:text-[var(--color-text)]"
                        >
                            Quy trình
                        </a>
                        <a
                            href="#evidence"
                            className="hover:text-[var(--color-text)]"
                        >
                            Kết quả
                        </a>
                        <a
                            href="#metrics"
                            className="hover:text-[var(--color-text)]"
                        >
                            Chỉ số
                        </a>
                    </nav>
                    {!user && !loading && (
                        <Link
                            to="/login"
                            className="hidden text-sm font-semibold text-[var(--color-text-muted)] hover:text-[var(--color-text)] sm:inline"
                        >
                            Đăng nhập
                        </Link>
                    )}
                    <Link to={primaryCta.to}>
                        <Button size="sm" variant="primary">
                            {user ? "Hệ thống ALPR" : "Bắt đầu"}
                        </Button>
                    </Link>
                </div>
            </header>

            <section className="mx-auto grid max-w-7xl gap-8 px-4 py-16 sm:px-6 lg:grid-cols-[1.05fr_0.95fr] lg:items-center lg:py-20">
                <div>
                    <Badge tone="info">
                        ALPR Việt Nam · nhận dạng biển số xe chuyên dùng cho
                        video
                    </Badge>
                    <h1 className="mt-6 max-w-4xl text-4xl font-bold leading-tight text-[var(--color-text)] sm:text-5xl">
                        Hệ thống nhận dạng biển số Việt Nam từ các nguồn video
                    </h1>
                    <p className="mt-5 max-w-2xl text-lg leading-8 text-[var(--color-text-muted)]">
                        Tải video lên hoặc trích đoạn một cửa sổ sự kiện ngắn;
                        hệ thống xử lý pipeline ALPR, nhận dạng biển số, lưu độ
                        tin cậy, ảnh phương tiện, ảnh biển số và lịch sử phiên.
                    </p>
                    <div className="mt-8 flex flex-col gap-3 sm:flex-row">
                        <Link to={primaryCta.to}>
                            <Button variant="primary" size="lg">
                                {primaryCta.label}
                            </Button>
                        </Link>
                        <a href="#pipeline">
                            <Button variant="secondary" size="lg">
                                Xem quy trình
                            </Button>
                        </a>
                    </div>
                </div>

                <HeroProductPreview />
            </section>

            <section
                id="pipeline"
                className="border-y border-[var(--color-border)] bg-[var(--color-bg-elevated)]/70"
            >
                <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6">
                    <div className="mb-8 max-w-3xl">
                        <p className="section-label">Quy trình xử lý</p>
                        <h2 className="mt-3 text-2xl font-bold">
                            ALPR theo video và theo vết phương tiện
                        </h2>
                        <p className="mt-3 text-[var(--color-text-muted)]">
                            Luồng nhận dạng biển số được diễn ra như sau:
                        </p>
                    </div>
                    <div className="grid gap-3 lg:grid-cols-5">
                        {pipelineSteps.map(([title, text]) => (
                            <article
                                key={title}
                                className="rounded-[var(--radius-panel)] border border-[var(--color-border)] bg-black/10 p-4"
                            >
                                <p className="data-font text-xs font-bold text-cyan-100">
                                    {title}
                                </p>
                                <p className="mt-3 text-sm leading-6 text-[var(--color-text-muted)]">
                                    {text}
                                </p>
                            </article>
                        ))}
                    </div>
                </div>
            </section>

            <section
                id="evidence"
                className="mx-auto grid max-w-7xl gap-6 px-4 py-14 sm:px-6 lg:grid-cols-[0.8fr_1.2fr] lg:items-start"
            >
                <div>
                    <p className="section-label">Giao diện kết quả</p>
                    <h2 className="mt-3 text-2xl font-bold">
                        Giao diện cho thấy mô hình đã biết gì
                    </h2>
                    <p className="mt-3 text-[var(--color-text-muted)]">
                        Người dùng không cần tin vào một chuỗi biển số đơn lẻ.
                        Hệ thống hiển thị nguồn video, trạng thái kết quả, ảnh
                        đối chiếu, độ tin cậy theo ký tự, phiếu OCR, biển số bị
                        loại và lịch sử.
                    </p>
                </div>
                <div className="grid gap-3 md:grid-cols-3">
                    {surfaces.map(([title, text]) => (
                        <article
                            key={title}
                            className="rounded-[var(--radius-panel)] border border-[var(--color-border)] bg-[var(--color-surface)] p-5"
                        >
                            <h3 className="text-base font-bold">{title}</h3>
                            <p className="mt-3 text-sm leading-6 text-[var(--color-text-muted)]">
                                {text}
                            </p>
                        </article>
                    ))}
                </div>
            </section>

            <section id="metrics" className="bg-[var(--color-bg-elevated)]/70">
                <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6">
                    <div className="mb-8 max-w-3xl">
                        <p className="section-label">Chỉ số đo được</p>
                        <h2 className="mt-3 text-2xl font-bold">
                            Kết quả huấn luyện các mô hình
                        </h2>
                    </div>
                    <div className="overflow-hidden rounded-[var(--radius-panel)] border border-[var(--color-border)]">
                        <table className="w-full border-collapse text-left text-sm">
                            <thead className="bg-[var(--color-bg)] text-[var(--color-text-muted)]">
                                <tr>
                                    <th className="px-4 py-3 font-semibold">
                                        Hạng mục
                                    </th>
                                    <th className="px-4 py-3 font-semibold">
                                        Giá trị
                                    </th>
                                    <th className="px-4 py-3 font-semibold">
                                        Ngữ cảnh
                                    </th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-[var(--color-border)] bg-[var(--color-surface)]">
                                {metrics.map(([name, value, context]) => (
                                    <tr key={name}>
                                        <td className="px-4 py-3 font-semibold">
                                            {name}
                                        </td>
                                        <td className="data-font px-4 py-3 text-cyan-100">
                                            {value}
                                        </td>
                                        <td className="px-4 py-3 text-[var(--color-text-muted)]">
                                            {context}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </section>

            <section className="mx-auto max-w-7xl px-4 py-14 sm:px-6">
                <div className="surface-panel flex flex-col gap-5 p-6 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                        <p className="section-label">Trải nghiệm hệ thống</p>
                        <h2 className="mt-2 text-2xl font-bold">
                            Chạy một video và xem kết quả hoàn chỉnh.
                        </h2>
                    </div>
                    <Link to={primaryCta.to}>
                        <Button variant="primary" size="lg">
                            {primaryCta.label}
                        </Button>
                    </Link>
                </div>
            </section>
        </main>
    );
}

function HeroProductPreview() {
    return (
        <div className="surface-panel overflow-hidden">
            <div className="panel-header">
                <div>
                    <p className="section-label">Xem trước hệ thống</p>
                    <p className="mt-1 text-sm font-semibold">
                        job_2026_0613 · đang xử lý
                    </p>
                </div>
                <Badge tone="success">4 biển số</Badge>
            </div>
            <div className="grid gap-px bg-[var(--color-border)] md:grid-cols-[1.25fr_0.75fr]">
                <div className="bg-black p-4">
                    <div className="relative aspect-video rounded-lg border border-cyan-300/30 bg-[var(--color-bg)] scanline-bg">
                        <div className="absolute left-[18%] top-[24%] h-[42%] w-[58%] rounded border border-cyan-300" />
                        <div className="absolute bottom-[26%] left-[34%] rounded border border-cyan-100 bg-cyan-300 px-3 py-1 plate-font text-xs font-bold tracking-widest text-black">
                            29A-678.90
                        </div>
                        <div className="absolute bottom-3 left-3 rounded-lg bg-black/70 px-3 py-2 text-xs text-cyan-50">
                            theo vết #12 · OCR hợp lệ · 92%
                        </div>
                    </div>
                </div>
                <div className="space-y-3 bg-[var(--color-bg-elevated)] p-4">
                    {["29A-678.90", "30A-123.45", "51F-888.88"].map(
                        (plate, index) => (
                            <div
                                key={plate}
                                className="rounded-lg border border-[var(--color-border)] bg-black/15 p-3"
                            >
                                <div className="flex items-center justify-between">
                                    <span className="plate-font text-sm font-bold tracking-wider">
                                        {plate}
                                    </span>
                                    <Badge
                                        tone={index === 0 ? "success" : "info"}
                                    >
                                        {index === 0 ? "92%" : "đang xử lí"}
                                    </Badge>
                                </div>
                                <p className="mt-2 text-xs text-[var(--color-text-muted)]">
                                    ảnh biển số · đã lưu bộ đệm theo vết
                                </p>
                            </div>
                        ),
                    )}
                </div>
            </div>
        </div>
    );
}
