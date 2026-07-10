export const LOGO_SVG_SRC = "/brand/logo.svg";
export const LOGO_PNG_SRC = "/brand/logo.png";

export default function BrandLogo({ className = "" }) {
    const handleFallback = (event) => {
        const image = event.currentTarget;
        if (image.dataset.logoFallback === "png") return;
        image.dataset.logoFallback = "png";
        image.src = LOGO_PNG_SRC;
    };

    return (
        <img
            src={LOGO_SVG_SRC}
            alt=""
            aria-hidden="true"
            className={["brand-logo", className].filter(Boolean).join(" ")}
            onError={handleFallback}
        />
    );
}
