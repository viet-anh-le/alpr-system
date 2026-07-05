import { useRef, useState } from "react";

import { Button, SegmentedControl, Select, TextInput } from "../ui";
import {
    OCR_OPTIONS,
    PREPROCESS_OPTIONS,
    formatBytes,
} from "../workbench/constants";

export default function SourceSelector({
    onConnectLive,
    onSelectFile,
    isConnectingLive = false,
    isOpeningVideo = false,
}) {
    const inputRef = useRef(null);
    const [tab, setTab] = useState("rtsp");
    const [url, setUrl] = useState("");
    const [file, setFile] = useState(null);
    const [preprocessMode, setPreprocessMode] = useState("none");
    const [ocrBackend, setOcrBackend] = useState("default");

    return (
        <section className="surface-panel overflow-hidden">
            <div className="panel-header">
                <div>
                    <p className="section-label">Nguồn sự kiện</p>
                    <h2 className="mt-1 text-lg font-bold">
                        Phân tích trích đoạn
                    </h2>
                </div>
                <SegmentedControl
                    value={tab}
                    onChange={setTab}
                    options={[
                        { value: "rtsp", label: "Video Streaming" },
                        { value: "upload", label: "Tải video" },
                    ]}
                />
            </div>

            <div className="p-4">
                {tab === "rtsp" ? (
                    <form
                        onSubmit={(event) => {
                            event.preventDefault();
                            if (url.trim())
                                onConnectLive(url.trim(), ocrBackend);
                        }}
                        className="grid gap-3 lg:grid-cols-[1fr_220px_auto] lg:items-end"
                    >
                        <label className="block space-y-1.5">
                            <span className="text-xs font-semibold text-[var(--color-text-muted)]">
                                URL RTSP
                            </span>
                            <TextInput
                                value={url}
                                onChange={(event) => setUrl(event.target.value)}
                                placeholder="rtsp://10.0.0.5:554/main"
                            />
                        </label>
                        <Select
                            label="Bộ OCR"
                            value={ocrBackend}
                            onChange={(event) =>
                                setOcrBackend(event.target.value)
                            }
                            options={OCR_OPTIONS}
                        />
                        <Button
                            type="submit"
                            variant="primary"
                            disabled={!url.trim()}
                            loading={isConnectingLive}
                        >
                            Kết nối
                        </Button>
                    </form>
                ) : (
                    <div className="space-y-4">
                        <button
                            type="button"
                            onClick={() => inputRef.current?.click()}
                            className="w-full rounded-[var(--radius-panel)] border border-dashed border-[var(--color-border)] bg-black/10 p-6 text-center transition-colors hover:border-[var(--color-border-strong)] hover:bg-white/5"
                        >
                            <p className="font-semibold">
                                Chọn video để đánh dấu đoạn
                            </p>
                            <p className="mt-1 text-sm text-[var(--color-text-muted)]">
                                Video dài sẽ được phát trong trình duyệt; chỉ
                                đoạn được chọn được gửi đi phân tích.
                            </p>
                        </button>
                        <input
                            ref={inputRef}
                            type="file"
                            accept="video/*"
                            className="hidden"
                            onChange={(event) =>
                                setFile(event.target.files?.[0] || null)
                            }
                        />
                        {file && (
                            <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-4">
                                <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
                                    <div className="min-w-0 flex-1">
                                        <p className="truncate text-sm font-semibold">
                                            {file.name}
                                        </p>
                                        <p className="mt-1 text-xs text-[var(--color-text-subtle)]">
                                            {formatBytes(file.size)}
                                        </p>
                                    </div>
                                    <Select
                                        label="Tiền xử lý"
                                        value={preprocessMode}
                                        onChange={(event) =>
                                            setPreprocessMode(
                                                event.target.value,
                                            )
                                        }
                                        options={PREPROCESS_OPTIONS}
                                        className="lg:w-52"
                                    />
                                    <Select
                                        label="Bộ OCR"
                                        value={ocrBackend}
                                        onChange={(event) =>
                                            setOcrBackend(event.target.value)
                                        }
                                        options={OCR_OPTIONS}
                                        className="lg:w-52"
                                    />
                                    <Button
                                        variant="primary"
                                        loading={isOpeningVideo}
                                        onClick={() =>
                                            onSelectFile(
                                                file,
                                                preprocessMode,
                                                ocrBackend,
                                            )
                                        }
                                    >
                                        {isOpeningVideo
                                            ? "Đang mở video"
                                            : "Mở video"}
                                    </Button>
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </section>
    );
}
