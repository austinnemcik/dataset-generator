type ToggleFieldProps = {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  description?: string;
  disabled?: boolean;
  tone?: "default" | "danger";
  compact?: boolean;
  className?: string;
};

export function ToggleField({
  checked,
  onChange,
  label,
  description,
  disabled = false,
  tone = "default",
  compact = false,
  className = "",
}: ToggleFieldProps) {
  const classes = [
    "toggle-field",
    compact ? "toggle-field-compact" : "",
    tone === "danger" ? "toggle-field-danger" : "",
    disabled ? "toggle-field-disabled" : "",
    className,
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <label className={classes}>
      <input
        checked={checked}
        className="toggle-field-input"
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
        type="checkbox"
      />
      <span className="toggle-field-copy">
        <span className="toggle-field-label">{label}</span>
        {description ? <span className="toggle-field-description">{description}</span> : null}
      </span>
      <span className="toggle-field-switch" aria-hidden="true">
        <span className="toggle-field-thumb" />
      </span>
    </label>
  );
}
