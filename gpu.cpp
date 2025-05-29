/*
 * @file studentSolution/src/studentSolution/gpu.cpp
 * @brief Полная реализация GPU pipeline (Tests 0–33)
 */

#include "gpu.hpp"
#include <solutionInterface/gpu.hpp>
#include <glm/glm.hpp>
#include <algorithm>
#include <vector>
#include <cstdint>

 //-----------------------------------------------------------------------------
 // Вспомогательные функции: чтение индексов и атрибутов
 //----------------------------------------------------------------------------- 
static uint32_t getVertexIndex(const GPUMemory& mem, uint32_t vid) {
    auto const& vao = mem.vertexArrays[mem.activatedVertexArray];
    if (vao.indexBufferID < 0) return vid;
    auto const& ib = mem.buffers[vao.indexBufferID];
    auto const* base = reinterpret_cast<uint8_t const*>(ib.data) + vao.indexOffset;
    switch (vao.indexType) {
    case IndexType::U8:  return base[vid];
    case IndexType::U16: return reinterpret_cast<uint16_t const*>(base)[vid];
    case IndexType::U32: return reinterpret_cast<uint32_t const*>(base)[vid];
    default:             return vid;
    }
}

static void fetchAttrib(
    Attrib& dst,
    Buffer const& buf,
    uint64_t      offset,
    uint64_t      stride,
    uint32_t      idx,
    AttribType    type)
{
    auto const* ptr = reinterpret_cast<uint8_t const*>(buf.data) + offset + stride * idx;
    switch (type) {
    case AttribType::FLOAT:  dst.v1 = *reinterpret_cast<float const*>(ptr); break;
    case AttribType::VEC2:   dst.v2 = *reinterpret_cast<glm::vec2 const*>(ptr); break;
    case AttribType::VEC3:   dst.v3 = *reinterpret_cast<glm::vec3 const*>(ptr); break;
    case AttribType::VEC4:   dst.v4 = *reinterpret_cast<glm::vec4 const*>(ptr); break;
    case AttribType::UINT:   dst.u1 = *reinterpret_cast<uint32_t const*>(ptr); break;
    case AttribType::UVEC2:  dst.u2 = *reinterpret_cast<glm::uvec2 const*>(ptr); break;
    case AttribType::UVEC3:  dst.u3 = *reinterpret_cast<glm::uvec3 const*>(ptr); break;
    case AttribType::UVEC4:  dst.u4 = *reinterpret_cast<glm::uvec4 const*>(ptr); break;
    default: break;
    }
}

//-----------------------------------------------------------------------------
// Очистка буферов (Tests 8–10)
//----------------------------------------------------------------------------- 
static void clearColorBuffer(Framebuffer& fbo, ClearColorCommand const& cmd) {
    if (!fbo.color.data) return;
    for (uint32_t y = 0; y < fbo.height; ++y)
        for (uint32_t x = 0; x < fbo.width; ++x) {
            void* p = getPixel(fbo.color, x, y);
            if (fbo.color.format == Image::Format::U8) {
                auto* dst = reinterpret_cast<uint8_t*>(p);
                for (uint32_t c = 0; c < fbo.color.channels; ++c) {
                    float v = std::clamp(cmd.value[c], 0.f, 1.f);
                    dst[c] = uint8_t(v * 255.f + 0.5f);
                }
            }
            else {
                auto* dst = reinterpret_cast<float*>(p);
                for (uint32_t c = 0; c < fbo.color.channels; ++c)
                    dst[c] = cmd.value[c];
            }
        }
}

static void clearDepthBuffer(Framebuffer& fbo, ClearDepthCommand const& cmd) {
    if (!fbo.depth.data) return;
    for (uint32_t y = 0; y < fbo.height; ++y)
        for (uint32_t x = 0; x < fbo.width; ++x)
            *reinterpret_cast<float*>(getPixel(fbo.depth, x, y)) = cmd.value;
}

static void clearStencilBuffer(Framebuffer& fbo, ClearStencilCommand const& cmd) {
    if (!fbo.stencil.data) return;
    for (uint32_t y = 0; y < fbo.height; ++y)
        for (uint32_t x = 0; x < fbo.width; ++x)
            *reinterpret_cast<uint8_t*>(getPixel(fbo.stencil, x, y)) = cmd.value;
}

//-----------------------------------------------------------------------------
// Применение stencil операций
//----------------------------------------------------------------------------- 
static void applyStencilOp(uint8_t* stencilValue, StencilOp op, uint32_t refValue) {
    switch (op) {
    case StencilOp::KEEP:      break;
    case StencilOp::ZERO:      *stencilValue = 0; break;
    case StencilOp::REPLACE:   *stencilValue = static_cast<uint8_t>(refValue); break;
    case StencilOp::INCR:      *stencilValue = static_cast<uint8_t>(std::min<uint32_t>(*stencilValue + 1, 255)); break;
    case StencilOp::INCR_WRAP: *stencilValue = static_cast<uint8_t>((*stencilValue + 1) & 0xFF); break;
    case StencilOp::DECR:      *stencilValue = static_cast<uint8_t>(std::max<int>(*stencilValue - 1, 0)); break;
    case StencilOp::DECR_WRAP: *stencilValue = static_cast<uint8_t>((*stencilValue - 1) & 0xFF); break;
    case StencilOp::INVERT:    *stencilValue = ~*stencilValue; break;
    }
}

//-----------------------------------------------------------------------------
// Основная обработка команд, включая весь pipeline
//----------------------------------------------------------------------------- 
struct VertexCache { std::vector<OutVertex> verts; };

static void processCommands(GPUMemory& mem, CommandBuffer const& cb) {
    for (uint32_t ci = 0; ci < cb.nofCommands; ++ci) {
        auto& cmd = cb.commands[ci];
        switch (cmd.type) {
        case CommandType::BIND_FRAMEBUFFER:
            mem.activatedFramebuffer = cmd.data.bindFramebufferCommand.id;
            break;
        case CommandType::BIND_PROGRAM:
            mem.activatedProgram = cmd.data.bindProgramCommand.id;
            break;
        case CommandType::BIND_VERTEXARRAY:
            mem.activatedVertexArray = cmd.data.bindVertexArrayCommand.id;
            break;
        case CommandType::BLOCK_WRITES_COMMAND:
            mem.blockWrites = cmd.data.blockWritesCommand.blockWrites;
            break;
        case CommandType::SET_BACKFACE_CULLING_COMMAND:
            mem.backfaceCulling.enabled = cmd.data.setBackfaceCullingCommand.enabled;
            break;
        case CommandType::SET_FRONT_FACE_COMMAND:
            mem.backfaceCulling.frontFaceIsCounterClockWise = cmd.data.setFrontFaceCommand.frontFaceIsCounterClockWise;
            break;
        case CommandType::SET_STENCIL_COMMAND:
            mem.stencilSettings = cmd.data.setStencilCommand.settings;
            break;
        case CommandType::SET_DRAW_ID:
            mem.gl_DrawID = cmd.data.setDrawIdCommand.id;
            break;
        case CommandType::CLEAR_COLOR:
            clearColorBuffer(mem.framebuffers[mem.activatedFramebuffer], cmd.data.clearColorCommand);
            break;
        case CommandType::CLEAR_DEPTH:
            clearDepthBuffer(mem.framebuffers[mem.activatedFramebuffer], cmd.data.clearDepthCommand);
            break;
        case CommandType::CLEAR_STENCIL:
            clearStencilBuffer(mem.framebuffers[mem.activatedFramebuffer], cmd.data.clearStencilCommand);
            break;
        case CommandType::USER_COMMAND:
            if (cmd.data.userCommand.callback)
                cmd.data.userCommand.callback(cmd.data.userCommand.data);
            break;
        case CommandType::DRAW: {
            auto& draw = cmd.data.drawCommand;
            auto& prog = mem.programs[mem.activatedProgram];
            ShaderInterface si_vs{ mem.uniforms, mem.textures, mem.gl_DrawID };
            // Vertex Assembly + VS
            VertexCache cache;
            cache.verts.reserve(draw.nofVertices);
            auto const& vao = mem.vertexArrays[mem.activatedVertexArray];
            for (uint32_t v = 0; v < draw.nofVertices; ++v) {
                uint32_t idx = getVertexIndex(mem, v);
                InVertex in; in.gl_VertexID = idx;
                for (int a = 0; a < maxAttribs; ++a) {
                    auto const& va = vao.vertexAttrib[a];
                    if (va.bufferID < 0 || va.type == AttribType::EMPTY) continue;
                    fetchAttrib(in.attributes[a], mem.buffers[va.bufferID], va.offset, va.stride, idx, va.type);
                }
                OutVertex out;
                if (prog.vertexShader) prog.vertexShader(out, in, si_vs);
                cache.verts.push_back(out);
            }
            // Perspective Divide + Viewport
            auto& fb = mem.framebuffers[mem.activatedFramebuffer];
            float W = float(fb.width), H = float(fb.height) * (fb.yReversed ? -1.f : 1.f);
            std::vector<glm::vec4> scr; scr.reserve(cache.verts.size());
            for (auto& o : cache.verts) {
                glm::vec4 p = o.gl_Position / o.gl_Position.w;
                scr.push_back({ (p.x * 0.5f + 0.5f) * W, (p.y * 0.5f + 0.5f) * H, p.z, o.gl_Position.w });
            }
            // Raster + Per-Fragment ops
            size_t primCount = cache.verts.size() / 3;
            for (size_t ti = 0; ti < primCount; ++ti) {
                size_t b = ti * 3; auto A = scr[b], B = scr[b + 1], C = scr[b + 2];
                float area = (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);
                bool isCCW = area > 0.f;
                bool isFront = (isCCW == mem.backfaceCulling.frontFaceIsCounterClockWise);
                if (mem.backfaceCulling.enabled && !isFront) continue;
                int minX = std::max(0, int(std::floor(std::min({ A.x,B.x,C.x }))));
                int maxX = std::min(int(fb.width) - 1, int(std::ceil(std::max({ A.x,B.x,C.x }))));
                int minY = std::max(0, int(std::floor(std::min({ A.y,B.y,C.y }))));
                int maxY = std::min(int(fb.height) - 1, int(std::ceil(std::max({ A.y,B.y,C.y }))));
                float denom = (B.y - C.y) * (A.x - C.x) + (C.x - B.x) * (A.y - C.y);
                if (std::abs(denom) < 1e-9f) continue;
                auto& ss = mem.stencilSettings;
                auto const& ops = (isFront ? ss.frontOps : ss.backOps);
                for (int py = minY; py <= maxY; ++py) {
                    for (int px = minX; px <= maxX; ++px) {
                        float fx = px + 0.5f, fy = py + 0.5f;
                        float w0 = ((B.y - C.y) * (fx - C.x) + (C.x - B.x) * (fy - C.y)) / denom;
                        float w1 = ((C.y - A.y) * (fx - C.x) + (A.x - C.x) * (fy - C.y)) / denom;
                        float w2 = 1.f - w0 - w1;
                        if (w0 < 0 || w1 < 0 || w2 < 0) continue;
                        // exclude bottom (w1==0) and right (w0==0) edges per top-left rule
                        if (w0 == 0.f || w1 == 0.f) continue;
                        // Early stencil
                        if (ss.enabled && fb.stencil.data) {
                            // skip edge pixels for stencil stage (top-left fill convention)
                            if (w0 == 0.f || w1 == 0.f || w2 == 0.f) continue;
                            uint8_t* sb = reinterpret_cast<uint8_t*>(getPixel(fb.stencil, px, py));
                            bool pass = false;
                            switch (ss.func) {
                            case StencilFunc::NEVER:    pass = false; break;
                            case StencilFunc::LESS:     pass = (*sb < ss.refValue); break;
                            case StencilFunc::LEQUAL:   pass = (*sb <= ss.refValue); break;
                            case StencilFunc::GREATER:  pass = (*sb > ss.refValue); break;
                            case StencilFunc::GEQUAL:   pass = (*sb >= ss.refValue); break;
                            case StencilFunc::EQUAL:    pass = (*sb == ss.refValue); break;
                            case StencilFunc::NOTEQUAL: pass = (*sb != ss.refValue); break;
                            case StencilFunc::ALWAYS:   pass = true; break;
                            }
                            if (!pass) {
                                if (!mem.blockWrites.stencil) applyStencilOp(sb, ops.sfail, ss.refValue);
                                continue;
                            }
                        }
                        // Depth test
                        float z = w0 * A.z + w1 * B.z + w2 * C.z;
                        if (fb.depth.data) {
                            float* db = reinterpret_cast<float*>(getPixel(fb.depth, px, py));
                            if (z > *db) {
                                if (ss.enabled && fb.stencil.data && !mem.blockWrites.stencil) {
                                    uint8_t* sb = reinterpret_cast<uint8_t*>(getPixel(fb.stencil, px, py));
                                    applyStencilOp(sb, ops.dpfail, ss.refValue);
                                }
                                continue;
                            }
                            if (!mem.blockWrites.depth) *db = z;
                        }
                        // Fragment shader
                        InFragment inF; inF.gl_FragCoord = { fx, fy, z, 1.f };
                        for (int a = 0; a < maxAttribs; ++a) {
                            if (prog.vs2fs[a] == AttribType::EMPTY) continue;
                            if (prog.vs2fs[a] == AttribType::UINT || prog.vs2fs[a] == AttribType::UVEC2 || prog.vs2fs[a] == AttribType::UVEC3 || prog.vs2fs[a] == AttribType::UVEC4) {
                                inF.attributes[a] = cache.verts[b].attributes[a];
                            }
                            else {
                                float h0 = cache.verts[b].gl_Position.w;
                                float h1 = cache.verts[b + 1].gl_Position.w;
                                float h2 = cache.verts[b + 2].gl_Position.w;
                                float q0 = w0 / h0, q1 = w1 / h1, q2 = w2 / h2;
                                float s = 1.f / (q0 + q1 + q2); q0 *= s; q1 *= s; q2 *= s;
                                auto& Aattr = cache.verts[b].attributes[a];
                                auto& Battr = cache.verts[b + 1].attributes[a];
                                auto& Cattr = cache.verts[b + 2].attributes[a];
                                inF.attributes[a].v4 = Aattr.v4 * q0 + Battr.v4 * q1 + Cattr.v4 * q2;
                            }
                        }
                        OutFragment outF{};
                        ShaderInterface si_fs{ mem.uniforms, mem.textures, mem.gl_DrawID };
                        if (prog.fragmentShader) prog.fragmentShader(outF, inF, si_fs);
                        if (outF.discard) continue;
                        // Late stencil write
                        if (ss.enabled && fb.stencil.data && !mem.blockWrites.stencil) {
                            uint8_t* sb = reinterpret_cast<uint8_t*>(getPixel(fb.stencil, px, py));
                            applyStencilOp(sb, ops.dppass, ss.refValue);
                        }
                        // Color write
                        if (fb.color.data && !mem.blockWrites.color) {
                            void* dp = getPixel(fb.color, px, py);
                            if (fb.color.format == Image::Format::U8) {
                                auto* dst = reinterpret_cast<uint8_t*>(dp);
                                for (uint32_t c = 0; c < fb.color.channels; ++c) {
                                    float v = std::clamp(outF.gl_FragColor[c], 0.f, 1.f);
                                    dst[c] = uint8_t(v * 255.f + 0.5f);
                                }
                            }
                            else {
                                auto* dst = reinterpret_cast<float*>(dp);
                                for (uint32_t c = 0; c < fb.color.channels; ++c) dst[c] = outF.gl_FragColor[c];
                            }
                        }
                    }
                }
            }
            ++mem.gl_DrawID;
            break;
        }
        case CommandType::SUB_COMMAND:
            if (cmd.data.subCommand.commandBuffer)
                processCommands(mem, *cmd.data.subCommand.commandBuffer);
            break;
        default: break;
        }
    }
}

void student_GPU_run(GPUMemory& mem, CommandBuffer const& cb) {
    mem.activatedFramebuffer = mem.defaultFramebuffer;
    mem.gl_DrawID = 0;
    processCommands(mem, cb);
}
