# shellcheck shell=bash
#
# Shared command-line parsing for host-validation scripts.
#
# This file is a library, not a validation script: source it, declare the
# script's options, then parse the arguments before any side effect.
#
#   source "$ROOT/mlx-host-validation/scripts/lib/cli.sh"
#   cli_describe "Validate native streaming through the public gateway."
#   cli_option PORT --port MLX_STREAMING_PORT port 18190 "Gateway port."
#   cli_option CAPTURE --metal-capture MLX_STREAMING_CAPTURE bool 0 "Record a capture."
#   cli_parse "$@"
#
# Each option resolves to: the command-line flag when given, otherwise the
# environment variable when set, otherwise the default. The resolved value is
# assigned to the named shell variable and printed as `option_<flag>=<value>`.
#
# Types:
#   string  any value
#   path    any non-empty value
#   port    integer in 1..65535
#   int     non-negative integer
#   number  non-negative decimal number
#   bool    0 or 1; `--flag` sets 1 and `--no-flag` sets 0
#   list    comma-separated values
#
# An empty default means "unset": the script derives the value after parsing.
#
# Every script also accepts:
#   -h, --help      print the options and exit 0
#   --print-config  print the resolved options and exit 0
#
# Usage errors exit 2. The library is compatible with macOS bash 3.2.

_CLI_VARS=()
_CLI_FLAGS=()
_CLI_ENVS=()
_CLI_TYPES=()
_CLI_DEFAULTS=()
_CLI_HELPS=()
_CLI_SUMMARY=""
_CLI_EXAMPLES=()

cli_describe() {
    _CLI_SUMMARY="$1"
}

cli_example() {
    _CLI_EXAMPLES+=("$1")
}

cli_option() {
    if [[ $# -ne 6 ]]; then
        echo "cli_option: expected VAR FLAG ENV TYPE DEFAULT HELP, got $# arguments" >&2
        exit 2
    fi
    case "$4" in
        string | path | port | int | number | bool | list) ;;
        *)
            echo "cli_option: unknown type '$4' for $2" >&2
            exit 2
            ;;
    esac
    _CLI_VARS+=("$1")
    _CLI_FLAGS+=("$2")
    _CLI_ENVS+=("$3")
    _CLI_TYPES+=("$4")
    _CLI_DEFAULTS+=("$5")
    _CLI_HELPS+=("$6")
}

cli_usage() {
    local script="${0#"${ROOT:-}"/}"
    local index flag metavar default
    echo "Usage: bash $script [options]"
    if [[ -n "$_CLI_SUMMARY" ]]; then
        echo
        echo "$_CLI_SUMMARY"
    fi
    echo
    echo "Options (flag > environment variable > default):"
    index=0
    while [[ $index -lt ${#_CLI_FLAGS[@]} ]]; do
        flag="${_CLI_FLAGS[$index]}"
        case "${_CLI_TYPES[$index]}" in
            bool) metavar="" ;;
            port) metavar=" PORT" ;;
            int | number) metavar=" N" ;;
            path) metavar=" PATH" ;;
            list) metavar=" A,B,..." ;;
            *) metavar=" VALUE" ;;
        esac
        if [[ "${_CLI_TYPES[$index]}" == "bool" ]]; then
            printf '  %s, --no-%s\n' "$flag" "${flag#--}"
        else
            printf '  %s%s\n' "$flag" "$metavar"
        fi
        default="${_CLI_DEFAULTS[$index]}"
        printf '      %s\n' "${_CLI_HELPS[$index]}"
        printf '      env %s; default: %s\n' "${_CLI_ENVS[$index]}" "${default:-(derived)}"
        index=$((index + 1))
    done
    echo "  --print-config"
    echo "      Print the resolved options and exit."
    echo "  -h, --help"
    echo "      Show this help and exit."
    if [[ ${#_CLI_EXAMPLES[@]} -gt 0 ]]; then
        echo
        echo "Examples:"
        for example in "${_CLI_EXAMPLES[@]}"; do
            echo "  $example"
        done
    fi
}

_cli_fail() {
    echo "error: $1" >&2
    echo "Run with --help to list the options." >&2
    exit 2
}

_cli_index_of() {
    local wanted="$1" index=0
    while [[ $index -lt ${#_CLI_FLAGS[@]} ]]; do
        if [[ "${_CLI_FLAGS[$index]}" == "$wanted" ]]; then
            echo "$index"
            return 0
        fi
        index=$((index + 1))
    done
    return 1
}

_cli_validate() {
    local flag="$1" type="$2" value="$3"
    if [[ -z "$value" ]]; then
        if [[ "$type" == "path" || "$type" == "port" ]]; then
            _cli_fail "$flag requires a non-empty value"
        fi
        return 0
    fi
    case "$type" in
        port)
            [[ "$value" =~ ^[0-9]+$ ]] && ((10#$value >= 1 && 10#$value <= 65535)) ||
                _cli_fail "$flag must be a TCP port in 1..65535, got '$value'"
            ;;
        int)
            [[ "$value" =~ ^[0-9]+$ ]] || _cli_fail "$flag must be a non-negative integer, got '$value'"
            ;;
        number)
            [[ "$value" =~ ^[0-9]+(\.[0-9]+)?$ || "$value" =~ ^\.[0-9]+$ ]] ||
                _cli_fail "$flag must be a non-negative number, got '$value'"
            ;;
        bool)
            [[ "$value" == "0" || "$value" == "1" ]] || _cli_fail "$flag must be 0 or 1, got '$value'"
            ;;
    esac
}

cli_print_config() {
    local index name
    index=0
    while [[ $index -lt ${#_CLI_FLAGS[@]} ]]; do
        name="${_CLI_FLAGS[$index]#--}"
        name="${name//-/_}"
        printf 'option_%s=%s\n' "$name" "${!_CLI_VARS[$index]}"
        index=$((index + 1))
    done
}

cli_parse() {
    local count=${#_CLI_FLAGS[@]}
    local values=() given=() index arg flag value env_set print_only=0
    index=0
    while [[ $index -lt $count ]]; do
        given+=(0)
        values+=("")
        index=$((index + 1))
    done

    while [[ $# -gt 0 ]]; do
        arg="$1"
        shift
        case "$arg" in
            -h | --help)
                cli_usage
                exit 0
                ;;
            --print-config)
                print_only=1
                continue
                ;;
            --*=*)
                flag="${arg%%=*}"
                value="${arg#*=}"
                index="$(_cli_index_of "$flag")" || _cli_fail "unknown option $flag"
                ;;
            --no-*)
                flag="--${arg#--no-}"
                index="$(_cli_index_of "$flag")" || _cli_fail "unknown option $arg"
                [[ "${_CLI_TYPES[$index]}" == "bool" ]] || _cli_fail "unknown option $arg"
                value=0
                ;;
            --*)
                flag="$arg"
                index="$(_cli_index_of "$flag")" || _cli_fail "unknown option $flag"
                if [[ "${_CLI_TYPES[$index]}" == "bool" ]]; then
                    value=1
                else
                    [[ $# -gt 0 ]] || _cli_fail "$flag requires a value"
                    value="$1"
                    shift
                fi
                ;;
            *)
                _cli_fail "unexpected argument '$arg'"
                ;;
        esac
        values[index]="$value"
        given[index]=1
    done

    index=0
    while [[ $index -lt $count ]]; do
        if [[ "${given[$index]}" == "1" ]]; then
            value="${values[$index]}"
        else
            eval "env_set=\${${_CLI_ENVS[$index]}+x}"
            if [[ -n "$env_set" ]]; then
                value="${!_CLI_ENVS[$index]}"
            else
                value="${_CLI_DEFAULTS[$index]}"
            fi
        fi
        _cli_validate "${_CLI_FLAGS[$index]}" "${_CLI_TYPES[$index]}" "$value"
        printf -v "${_CLI_VARS[$index]}" '%s' "$value"
        index=$((index + 1))
    done

    if [[ "$print_only" == "1" ]]; then
        cli_print_config
        exit 0
    fi
    cli_print_config
}

cli_require_free_port() {
    local port="$1" flag="$2" pid
    command -v lsof >/dev/null 2>&1 || return 0
    pid="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -n 1 || true)"
    if [[ -n "$pid" ]]; then
        echo "error: port $port is already in use by pid $pid ($(ps -o comm= -p "$pid" 2>/dev/null)); choose another with $flag" >&2
        exit 1
    fi
}

cli_split_list() {
    IFS=',' read -r -a "$1" <<<"$2"
}
